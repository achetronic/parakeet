// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0
//
// Minimal cgo shim that enables the ONNX Runtime ROCm execution provider.
//
// The Go binding (yalue/onnxruntime_go) ships no ROCm wrapper, and ORT's
// generic name-based AppendExecutionProvider rejects "ROCM": its supported
// list covers DML/QNN/OpenVINO/... only. The ROCm EP must be enabled through
// a dedicated C API entry point. Instead of indexing the OrtApi function
// table (whose layout is version-sensitive), we call the legacy free function
//
//   OrtStatus* OrtSessionOptionsAppendExecutionProvider_ROCM(
//       OrtSessionOptions* options, int device_id);
//
// which ROCm-enabled libonnxruntime builds export directly (verified against
// onnxruntime_rocm 1.22.1, the last release shipping the ROCm EP). The
// legacy entry point takes only the device id; arena/MIOpen tuning keys are
// not configurable through it, so ORT's defaults apply.
//
// Error messages are read via OrtApi::GetErrorMessage, a fixed table entry:
// the OrtApi table begins with CreateStatus (0), GetErrorCode (1) and
// GetErrorMessage (2) in every API version.

#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>

typedef struct OrtStatus OrtStatus;

/* Enables the ROCm EP on session_options for the given device.
   Returns 0 on success, -1 on failure with a message in errbuf. */
int AppendROCmProvider(const char *lib_path, void *session_options,
                       int device_id, char *errbuf, size_t errbuf_len) {
  /* The Go binding already dlopened this same library; dlopen is refcounted,
     so this just returns the existing handle. */
  void *handle = dlopen(lib_path, RTLD_LAZY);
  if (handle == NULL) {
    snprintf(errbuf, errbuf_len, "load ONNX Runtime library %s: %s",
             lib_path, dlerror());
    return -1;
  }

  typedef OrtStatus *(*AppendRocmFn)(void *, int);
  AppendRocmFn append = (AppendRocmFn)dlsym(
      handle, "OrtSessionOptionsAppendExecutionProvider_ROCM");
  if (append == NULL) {
    snprintf(errbuf, errbuf_len,
             "ROCm execution provider not found in %s: symbol "
             "OrtSessionOptionsAppendExecutionProvider_ROCM is missing (is "
             "this a ROCm-enabled ONNX Runtime build?)",
             lib_path);
    return -1;
  }

  OrtStatus *status = append(session_options, device_id);
  if (status == NULL) {
    return 0;
  }

  /* Surface the provider's own error message through
     OrtGetApiBase()->GetApi(22)->GetErrorMessage (table index 2). */
  const char *msg = NULL;
  typedef const void *(*GetApiBaseFn)(void);
  typedef const void *(*GetApiFn)(unsigned int);
  typedef const char *(*GetErrorMessageFn)(const OrtStatus *);

  GetApiBaseFn get_api_base = (GetApiBaseFn)dlsym(handle, "OrtGetApiBase");
  if (get_api_base != NULL) {
    const void *const *api_base = (const void *const *)get_api_base();
    if (api_base != NULL) {
      GetApiFn get_api = (GetApiFn)api_base[0]; /* OrtApiBase::GetApi */
      const void *const *api =
          (const void *const *)get_api(22); /* binding targets ORT API 22 */
      if (api != NULL) {
        GetErrorMessageFn get_error_message = (GetErrorMessageFn)api[2];
        if (get_error_message != NULL) {
          msg = get_error_message(status);
        }
      }
    }
  }
  snprintf(errbuf, errbuf_len, "%s",
           msg != NULL ? msg : "unknown ROCm provider error");
  /* The OrtStatus is intentionally not released: its ReleaseStatus table
     index is not pinned by the minimal declarations above, and this path
     only runs once at startup. */
  return -1;
}
