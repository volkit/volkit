#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>

#include <vkt/ExecutionPolicy.h>

#include "common.h"

static void print(vktExecutionPolicy_t ep)
{
    char str[1024];
    char device[20];
    char hostApi[20];
    char deviceApi[20];
    uint64_t tid;

    pthread_threadid_np(0, &tid);

    if (ep.device == vktExecutionPolicyDeviceCPU)
        sprintf(device, "%s", "CPU");
    else
        sprintf(device, "%s", "GPU");

    if (ep.hostApi == vktExecutionPolicyHostAPISerial)
        sprintf(hostApi, "%s", "Serial");
    else
        sprintf(hostApi, "%s", "???");

    if (ep.deviceApi == vktExecutionPolicyDeviceAPICUDA)
        sprintf(deviceApi, "%s", "CUDA");
    else
        sprintf(deviceApi, "%s", "???");

    sprintf(str, "ExecutionPolicy thread %lld\n", tid);
    sprintf(str + strlen(str), "device .....: %s\n", device);
    sprintf(str + strlen(str), "hostApi ....: %s\n", hostApi);
    sprintf(str + strlen(str), "deviceApi ..: %s\n", deviceApi);
    fprintf(stdout, "\n%s\n", str);
}

void* threadFunc(void* arg)
{
    vktExecutionPolicy_t ep;

    (void)arg;

    // Create a default execution policy
    memset(&ep, 0, sizeof(ep));
    print(ep);

    // Set this as policy for this thread
    // (this will have no effect)
    vktSetThreadExecutionPolicy(ep);

    // Change to GPU
    ep.device = vktExecutionPolicyDeviceGPU;
    vktSetThreadExecutionPolicy(ep);

    // Changes were applied
    ep = vktGetThreadExecutionPolicy();
    print(ep);

    return NULL;
}

int main()
{
    vktExecutionPolicy_t ep;
    pthread_t thread;

    // Default execution policy (CPU)
    ep = vktGetThreadExecutionPolicy();
    print(ep);

    // Change execution policy in main thread
    ep.device = vktExecutionPolicyDeviceGPU;
    vktSetThreadExecutionPolicy(ep);

    // Changes were applied
    ep = vktGetThreadExecutionPolicy();
    print(ep);

    // New thread
    pthread_create(&thread, NULL, &threadFunc, NULL);
    pthread_join(thread, NULL);

    // Execution policy in main thread remains unchanged
    ep = vktGetThreadExecutionPolicy();
    print(ep);
}
