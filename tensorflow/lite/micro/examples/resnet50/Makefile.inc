resnet50_MODEL_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/resnet50/ic_model_settings.cc

resnet50_MODEL_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/resnet50/ic_model_settings.h

resnet50_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/resnet50/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/resnet50/ic_test.cc \
$(resnet50_MODEL_SRCS)

resnet50_TEST_HDRS := \
$(resnet50_MODEL_HDRS)

DETECTION_RESPONDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/resnet50/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/resnet50/detection_responder_test.cc

DETECTION_RESPONDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/resnet50/detection_responder.h

resnet50_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/resnet50/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/resnet50/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/resnet50/main.cc \
$(resnet50_MODEL_SRCS)

resnet50_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/resnet50/detection_responder.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/resnet50/image_provider.h \
$(resnet50_MODEL_HDRS)

resnet50_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/resnet50_model.tflite
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/dataset/*.bin \
# resnet50_GENERATOR_INPUTS := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/dataset/*.bin)

resnet50_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/resnet50_model.cc

resnet50_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/resnet50_model.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/resnet50/*/Makefile.inc)

# TODO(b/268568089): This test is taking very long time to finish; causing the
# CI to run for a long time to finish.
ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifimini hifi3z))
  # Tests loading and running a vision model.
  $(eval $(call microlite_test,resnet50_test,\
  $(resnet50_TEST_SRCS),$(resnet50_TEST_HDRS),$(resnet50_GENERATOR_INPUTS)))
endif

# Builds a standalone object recognition binary.
$(eval $(call microlite_test,resnet50,\
$(resnet50_SRCS),$(resnet50_HDRS),$(resnet50_GENERATOR_INPUTS)))

# Add sources and headers generated from $(resnet50_GENERATOR_INPUTS).
resnet50_SRCS += $(resnet50_GENERATED_SRCS)
resnet50_HDRS += $(resnet50_GENERATED_HDRS)

list_resnet50_example_sources:
	@echo $(resnet50_SRCS)

list_resnet50_example_headers:
	@echo $(resnet50_HDRS)
