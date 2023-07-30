anomaly_detection_MODEL_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/ad_model_settings.cc

anomaly_detection_MODEL_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/ad_model_settings.h

anomaly_detection_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/ad_test.cc \
$(anomaly_detection_MODEL_SRCS)

anomaly_detection_TEST_HDRS := \
$(anomaly_detection_MODEL_HDRS)

IMAGE_PROVIDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/image_provider_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/ad_model_settings.cc

IMAGE_PROVIDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/image_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/model_settings.h

DETECTION_RESPONDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/detection_responder_test.cc

DETECTION_RESPONDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/detection_responder.h

anomaly_detection_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/main.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/main_functions.cc \
$(anomaly_detection_MODEL_SRCS)

anomaly_detection_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/detection_responder.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/image_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/main_functions.h \
$(anomaly_detection_MODEL_HDRS)

anomaly_detection_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/ad_model.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/dataset/*.bin \
# anomaly_detection_GENERATOR_INPUTS := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/dataset/*.bin)

anomaly_detection_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/ad_model.cc

anomaly_detection_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/ad_model.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/*/Makefile.inc)

# TODO(b/268568089): This test is taking very long time to finish; causing the
# CI to run for a long time to finish.
ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifimini hifi3z))
  # Tests loading and running a vision model.
  $(eval $(call microlite_test,anomaly_detection_test,\
  $(anomaly_detection_TEST_SRCS),$(anomaly_detection_TEST_HDRS),$(anomaly_detection_GENERATOR_INPUTS)))
endif

# Tests the image provider module.
$(eval $(call microlite_test,image_provider_test,\
$(IMAGE_PROVIDER_TEST_SRCS),$(IMAGE_PROVIDER_TEST_HDRS)))

# Tests the detection responder module.
$(eval $(call microlite_test,detection_responder_test,\
$(DETECTION_RESPONDER_TEST_SRCS),$(DETECTION_RESPONDER_TEST_HDRS)))

# Builds a standalone object recognition binary.
$(eval $(call microlite_test,anomaly_detection,\
$(anomaly_detection_SRCS),$(anomaly_detection_HDRS),$(anomaly_detection_GENERATOR_INPUTS)))

# Add sources and headers generated from $(anomaly_detection_GENERATOR_INPUTS).
anomaly_detection_SRCS += $(anomaly_detection_GENERATED_SRCS)
anomaly_detection_HDRS += $(anomaly_detection_GENERATED_HDRS)

list_anomaly_detection_example_sources:
	@echo $(anomaly_detection_SRCS)

list_anomaly_detection_example_headers:
	@echo $(anomaly_detection_HDRS)
