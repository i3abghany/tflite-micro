densenet121_MODEL_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/densenet121/ic_model_settings.cc

densenet121_MODEL_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/densenet121/ic_model_settings.h

densenet121_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/densenet121/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/densenet121/ic_test.cc \
$(densenet121_MODEL_SRCS)

densenet121_TEST_HDRS := \
$(densenet121_MODEL_HDRS)

DETECTION_RESPONDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/densenet121/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/densenet121/detection_responder_test.cc

DETECTION_RESPONDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/densenet121/detection_responder.h

densenet121_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/densenet121/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/densenet121/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/densenet121/main.cc \
$(densenet121_MODEL_SRCS)

densenet121_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/densenet121/detection_responder.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/densenet121/image_provider.h \
$(densenet121_MODEL_HDRS)

densenet121_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/densenet121_model.tflite
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/dataset/*.bin \
# densenet121_GENERATOR_INPUTS := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/dataset/*.bin)

densenet121_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/densenet121_model.cc

densenet121_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/densenet121_model.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/densenet121/*/Makefile.inc)

# TODO(b/268568089): This test is taking very long time to finish; causing the
# CI to run for a long time to finish.
ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifimini hifi3z))
  # Tests loading and running a vision model.
  $(eval $(call microlite_test,densenet121_test,\
  $(densenet121_TEST_SRCS),$(densenet121_TEST_HDRS),$(densenet121_GENERATOR_INPUTS)))
endif

# Builds a standalone object recognition binary.
$(eval $(call microlite_test,densenet121,\
$(densenet121_SRCS),$(densenet121_HDRS),$(densenet121_GENERATOR_INPUTS)))

# Add sources and headers generated from $(densenet121_GENERATOR_INPUTS).
densenet121_SRCS += $(densenet121_GENERATED_SRCS)
densenet121_HDRS += $(densenet121_GENERATED_HDRS)

list_densenet121_example_sources:
	@echo $(densenet121_SRCS)

list_densenet121_example_headers:
	@echo $(densenet121_HDRS)
