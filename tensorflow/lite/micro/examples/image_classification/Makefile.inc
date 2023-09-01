image_classification_MODEL_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/ic_model_settings.cc

image_classification_MODEL_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/ic_model_settings.h

image_classification_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/ic_test.cc \
$(image_classification_MODEL_SRCS)

image_classification_TEST_HDRS := \
$(image_classification_MODEL_HDRS)

DETECTION_RESPONDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/detection_responder_test.cc

DETECTION_RESPONDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/detection_responder.h

image_classification_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/main.cc \
$(image_classification_MODEL_SRCS)

image_classification_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/detection_responder.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/image_provider.h \
$(image_classification_MODEL_HDRS)

image_classification_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/ic_model.tflite
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/dataset/*.bin
# image_classification_GENERATOR_INPUTS := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/dataset/*.bin)

image_classification_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/ic_model.cc

image_classification_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/ic_model.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/image_classification/*/Makefile.inc)

# TODO(b/268568089): This test is taking very long time to finish; causing the
# CI to run for a long time to finish.
ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifimini hifi3z))
  # Tests loading and running a vision model.
  $(eval $(call microlite_test,image_classification_test,\
  $(image_classification_TEST_SRCS),$(image_classification_TEST_HDRS),$(image_classification_GENERATOR_INPUTS)))
endif

# Builds a standalone object recognition binary.
$(eval $(call microlite_test,image_classification,\
$(image_classification_SRCS),$(image_classification_HDRS),$(image_classification_GENERATOR_INPUTS)))

# Add sources and headers generated from $(image_classification_GENERATOR_INPUTS).
image_classification_SRCS += $(image_classification_GENERATED_SRCS)
image_classification_HDRS += $(image_classification_GENERATED_HDRS)

list_image_classification_example_sources:
	@echo $(image_classification_SRCS)

list_image_classification_example_headers:
	@echo $(image_classification_HDRS)
