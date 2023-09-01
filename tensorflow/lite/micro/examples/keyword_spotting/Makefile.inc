meyword_spotting_MODEL_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/kws_model_settings.cc

keyword_spotting_MODEL_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/kws_model_settings.h

keyword_spotting_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/kws_test.cc \
$(keyword_spotting_MODEL_SRCS)

keyword_spotting_TEST_HDRS := \
$(keyword_spotting_MODEL_HDRS)

IMAGE_PROVIDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/image_provider_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/kws_model_settings.cc

IMAGE_PROVIDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/image_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/model_settings.h

keyword_spotting_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/main.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/main_functions.cc \
$(keyword_spotting_MODEL_SRCS)

keyword_spotting_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/detection_responder.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/image_provider.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/main_functions.h \
$(keyword_spotting_MODEL_HDRS)

keyword_spotting_GENERATOR_INPUTS := \
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/models/kws_ref_model.tflite \
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/dataset/tst_000595_On_5.bin \
# keyword_spotting_GENERATOR_INPUTS := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/dataset/*.bin)

keyword_spotting_GENERATED_SRCS := \
# $(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/kws_data.cc

keyword_spotting_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/kws_data.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/keyword_spotting/*/Makefile.inc)

# TODO(b/268568089): This test is taking very long time to finish; causing the
# CI to run for a long time to finish.
ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifimini hifi3z))
  # Tests loading and running a vision model.
  $(eval $(call microlite_test,keyword_spotting_test,\
  $(keyword_spotting_TEST_SRCS),$(keyword_spotting_TEST_HDRS),$(keyword_spotting_GENERATOR_INPUTS)))
endif

# Builds a standalone object recognition binary.
$(eval $(call microlite_test,keyword_spotting,\
$(keyword_spotting_SRCS),$(keyword_spotting_HDRS),$(keyword_spotting_GENERATOR_INPUTS)))

# Add sources and headers generated from $(keyword_spotting_GENERATOR_INPUTS).
keyword_spotting_SRCS += $(keyword_spotting_GENERATED_SRCS)
keyword_spotting_HDRS += $(keyword_spotting_GENERATED_HDRS)

list_keyword_spotting_example_sources:
	@echo $(keyword_spotting_SRCS)

list_keyword_spotting_example_headers:
	@echo $(keyword_spotting_HDRS)
