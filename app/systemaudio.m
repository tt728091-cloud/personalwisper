// systemaudio: streams the sound every app on this Mac is playing (a meeting's other voices, for example) to
// stdout as raw 16 kHz mono float32 samples, until it gets SIGTERM/SIGINT or stdin closes.
// Uses Core Audio process taps (macOS 14.2+). Nothing is written to disk.
// Built into PersonalWisper.app by build_app.sh, so macOS asks PersonalWisper for "System Audio Recording".
// Without that permission the tap delivers silence rather than an error, so the caller checks for that.
#import <AVFoundation/AVFoundation.h>
#import <CoreAudio/AudioHardwareTapping.h>
#import <CoreAudio/CATapDescription.h>
#import <CoreAudio/CoreAudio.h>
#import <Foundation/Foundation.h>
#include <signal.h>
#include <unistd.h>

static AudioObjectID tapID = kAudioObjectUnknown, deviceID = kAudioObjectUnknown;
static AudioDeviceIOProcID procID = NULL;
static dispatch_source_t stopSignals[2];

static void check(OSStatus status, const char *message) {
    if (status != noErr) {
        fprintf(stderr, "systemaudio: %s (%d)\n", message, (int)status);
        exit(1);
    }
}

static void get_property(AudioObjectID object, AudioObjectPropertySelector selector, UInt32 size, void *value) {
    AudioObjectPropertyAddress address = {selector, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    check(AudioObjectGetPropertyData(object, &address, 0, NULL, &size, value), "can't read an audio property");
}

// Always called on the main queue: stopping the device from its own audio callback would deadlock.
static void stop(int code) {
    if (procID) {
        AudioDeviceStop(deviceID, procID);
        AudioDeviceDestroyIOProcID(deviceID, procID);
    }
    if (deviceID != kAudioObjectUnknown) AudioHardwareDestroyAggregateDevice(deviceID);
    if (tapID != kAudioObjectUnknown) AudioHardwareDestroyProcessTap(tapID);
    exit(code);
}

int main(void) {
    @autoreleasepool {
        // A tap on everything this Mac plays, left audible.
        CATapDescription *tap = [[CATapDescription alloc] initStereoGlobalTapButExcludeProcesses:@[]];
        tap.UUID = [NSUUID UUID];
        tap.name = @"PersonalWisper meeting transcript";
        tap.privateTap = YES;
        tap.muteBehavior = CATapUnmuted;
        check(AudioHardwareCreateProcessTap(tap, &tapID), "can't create the audio tap");

        // The tap is read through a private aggregate device built around the current output device.
        AudioObjectID output = kAudioObjectUnknown;
        get_property(kAudioObjectSystemObject, kAudioHardwarePropertyDefaultSystemOutputDevice, sizeof output, &output);
        CFStringRef outputUID = NULL;
        get_property(output, kAudioDevicePropertyDeviceUID, sizeof outputUID, &outputUID);
        NSDictionary *aggregate = @{
            @kAudioAggregateDeviceNameKey: @"PersonalWisper meeting transcript",
            @kAudioAggregateDeviceUIDKey: [NSUUID UUID].UUIDString,
            @kAudioAggregateDeviceMainSubDeviceKey: (__bridge NSString *)outputUID,
            @kAudioAggregateDeviceIsPrivateKey: @YES,
            @kAudioAggregateDeviceIsStackedKey: @NO,
            @kAudioAggregateDeviceTapAutoStartKey: @YES,
            @kAudioAggregateDeviceSubDeviceListKey: @[@{@kAudioSubDeviceUIDKey: (__bridge NSString *)outputUID}],
            @kAudioAggregateDeviceTapListKey: @[@{@kAudioSubTapDriftCompensationKey: @YES,
                                                  @kAudioSubTapUIDKey: tap.UUID.UUIDString}],
        };
        check(AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef)aggregate, &deviceID),
              "can't create the aggregate device");

        AudioStreamBasicDescription tapFormat;
        get_property(tapID, kAudioTapPropertyFormat, sizeof tapFormat, &tapFormat);
        AVAudioFormat *inputFormat = [[AVAudioFormat alloc] initWithStreamDescription:&tapFormat];
        AVAudioFormat *outputFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                                                       sampleRate:16000 channels:1 interleaved:NO];
        AVAudioConverter *converter = [[AVAudioConverter alloc] initFromFormat:inputFormat toFormat:outputFormat];
        if (!inputFormat || !converter) {
            fprintf(stderr, "systemaudio: unsupported tap format\n");
            stop(1);
        }
        converter.downmix = YES;

        dispatch_queue_t queue = dispatch_queue_create("systemaudio", DISPATCH_QUEUE_SERIAL);
        check(AudioDeviceCreateIOProcIDWithBlock(&procID, deviceID, queue,
            ^(const AudioTimeStamp *now, const AudioBufferList *inputData, const AudioTimeStamp *inputTime,
              AudioBufferList *outputData, const AudioTimeStamp *outputTime) {
                AVAudioPCMBuffer *input = [[AVAudioPCMBuffer alloc] initWithPCMFormat:inputFormat
                                                                     bufferListNoCopy:inputData deallocator:nil];
                if (!input || input.frameLength == 0) return;
                AVAudioFrameCount capacity =
                    (AVAudioFrameCount)(input.frameLength * 16000.0 / inputFormat.sampleRate) + 32;
                AVAudioPCMBuffer *converted = [[AVAudioPCMBuffer alloc] initWithPCMFormat:outputFormat
                                                                            frameCapacity:capacity];
                __block BOOL supplied = NO;
                [converter convertToBuffer:converted error:nil
                        withInputFromBlock:^AVAudioBuffer *(AVAudioPacketCount count,
                                                            AVAudioConverterInputStatus *status) {
                    if (supplied) {
                        *status = AVAudioConverterInputStatus_NoDataNow;  // keeps the resampler's state
                        return nil;
                    }
                    supplied = YES;
                    *status = AVAudioConverterInputStatus_HaveData;
                    return input;
                }];
                size_t bytes = converted.frameLength * sizeof(float);
                if (bytes && write(STDOUT_FILENO, converted.floatChannelData[0], bytes) < 0) {
                    dispatch_async(dispatch_get_main_queue(), ^{ stop(0); });  // the app is gone
                }
            }), "can't read the aggregate device");

        signal(SIGPIPE, SIG_IGN);  // a closed stdout shows up as a failed write instead
        for (int i = 0; i < 2; i++) {
            int sig = i ? SIGINT : SIGTERM;
            signal(sig, SIG_IGN);
            stopSignals[i] = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, sig, 0, dispatch_get_main_queue());
            dispatch_source_set_event_handler(stopSignals[i], ^{ stop(0); });
            dispatch_resume(stopSignals[i]);
        }
        // The app holds our stdin open; if it goes away, so do we.
        [NSThread detachNewThreadWithBlock:^{
            char buffer[64];
            while (read(STDIN_FILENO, buffer, sizeof buffer) > 0) {}
            dispatch_async(dispatch_get_main_queue(), ^{ stop(0); });
        }];

        check(AudioDeviceStart(deviceID, procID), "can't start recording");
        fprintf(stderr, "systemaudio: recording\n");
    }
    dispatch_main();
}
