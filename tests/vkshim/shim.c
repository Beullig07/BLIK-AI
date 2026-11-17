// Test-only Vulkan ICD wrapper: forwards everything to Mesa lavapipe, but reports
// SwiftShader's vendor/device IDs so headless Chromium will hand out the adapter.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

typedef PFN_vkVoidFunction (*GIPA)(VkInstance, const char*);
typedef VkResult (*NEG)(uint32_t*);
static void* lib;
static GIPA real_gipa;
static GIPA real_pdpa;
static NEG real_neg;
static void (*real_props)(VkPhysicalDevice, VkPhysicalDeviceProperties*);
static void (*real_props2)(VkPhysicalDevice, VkPhysicalDeviceProperties2*);

static void load(void) {
  if (lib) return;
  lib = dlopen("/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so", RTLD_NOW | RTLD_LOCAL);
  real_gipa = (GIPA)dlsym(lib, "vk_icdGetInstanceProcAddr");
  real_pdpa = (GIPA)dlsym(lib, "vk_icdGetPhysicalDeviceProcAddr");
  real_neg = (NEG)dlsym(lib, "vk_icdNegotiateLoaderICDInterfaceVersion");
}
static void patch(VkPhysicalDeviceProperties* p) { p->vendorID = 0x1AE0; p->deviceID = 0xC0DE; }
static VKAPI_ATTR void VKAPI_CALL my_props(VkPhysicalDevice d, VkPhysicalDeviceProperties* p) { real_props(d, p); patch(p); }
static VKAPI_ATTR void VKAPI_CALL my_props2(VkPhysicalDevice d, VkPhysicalDeviceProperties2* p) { real_props2(d, p); patch(&p->properties); }

__attribute__((visibility("default"))) VkResult vk_icdNegotiateLoaderICDInterfaceVersion(uint32_t* v) { load(); return real_neg ? real_neg(v) : VK_SUCCESS; }
__attribute__((visibility("default"))) PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance i, const char* n) {
  load();
  PFN_vkVoidFunction f = real_gipa(i, n);
  if (!f) return f;
  if (!strcmp(n, "vkGetPhysicalDeviceProperties")) { real_props = (void*)f; return (PFN_vkVoidFunction)my_props; }
  if (!strcmp(n, "vkGetPhysicalDeviceProperties2") || !strcmp(n, "vkGetPhysicalDeviceProperties2KHR")) { real_props2 = (void*)f; return (PFN_vkVoidFunction)my_props2; }
  return f;
}
__attribute__((visibility("default"))) PFN_vkVoidFunction vk_icdGetPhysicalDeviceProcAddr(VkInstance i, const char* n) { load(); return real_pdpa ? real_pdpa(i, n) : NULL; }
