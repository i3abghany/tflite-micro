visual_wake_words_MODEL_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/vww_model_settings.cc

visual_wake_words_MODEL_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/vww_model_settings.h

visual_wake_words_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/vww_test.cc \
$(visual_wake_words_MODEL_SRCS)

visual_wake_words_TEST_HDRS := \
$(visual_wake_words_MODEL_HDRS)

IMAGE_PROVIDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/image_provider_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/vww_model_settings.cc

IMAGE_PROVIDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/image_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/model_settings.h

DETECTION_RESPONDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/detection_responder_test.cc

DETECTION_RESPONDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/detection_responder.h

visual_wake_words_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/main.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/main_functions.cc \
$(visual_wake_words_MODEL_SRCS)

visual_wake_words_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/detection_responder.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/image_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/main_functions.h \
$(visual_wake_words_MODEL_HDRS)

visual_wake_words_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/vww_model.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/dataset/*.bin \
# visual_wake_words_GENERATOR_INPUTS := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/dataset/*.bin)

visual_wake_words_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/vww_model.cc

visual_wake_words_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/vww_model.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/visual_wake_words/*/Makefile.inc)

# TODO(b/268568089): This test is taking very long time to finish; causing the
# CI to run for a long time to finish.
ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifimini hifi3z))
  # Tests loading and running a vision model.
  $(eval $(call microlite_test,visual_wake_words_test,\
  $(visual_wake_words_TEST_SRCS),$(visual_wake_words_TEST_HDRS),$(visual_wake_words_GENERATOR_INPUTS)))
endif

# Tests the image provider module.
$(eval $(call microlite_test,image_provider_test,\
$(IMAGE_PROVIDER_TEST_SRCS),$(IMAGE_PROVIDER_TEST_HDRS)))

# Tests the detection responder module.
$(eval $(call microlite_test,detection_responder_test,\
$(DETECTION_RESPONDER_TEST_SRCS),$(DETECTION_RESPONDER_TEST_HDRS)))

# Builds a standalone object recognition binary.
$(eval $(call microlite_test,visual_wake_words,\
$(visual_wake_words_SRCS),$(visual_wake_words_HDRS),$(visual_wake_words_GENERATOR_INPUTS)))

# Add sources and headers generated from $(visual_wake_words_GENERATOR_INPUTS).
visual_wake_words_SRCS += $(visual_wake_words_GENERATED_SRCS)
visual_wake_words_HDRS += $(visual_wake_words_GENERATED_HDRS)

list_visual_wake_words_example_sources:
	@echo $(visual_wake_words_SRCS)

list_visual_wake_words_example_headers:
	@echo $(visual_wake_words_HDRS)
