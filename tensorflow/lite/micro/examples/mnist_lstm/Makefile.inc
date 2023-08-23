mnist_lstm_MODEL_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/lstm_model_settings.cc

mnist_lstm_MODEL_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/lstm_model_settings.h

mnist_lstm_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/lstm_test.cc \
$(mnist_lstm_MODEL_SRCS)

mnist_lstm_TEST_HDRS := \
$(mnist_lstm_MODEL_HDRS)

DETECTION_RESPONDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/detection_responder_test.cc

DETECTION_RESPONDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/detection_responder.h

mnist_lstm_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/main.cc \
$(mnist_lstm_MODEL_SRCS)

mnist_lstm_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/detection_responder.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/image_provider.h \
$(mnist_lstm_MODEL_HDRS)

mnist_lstm_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/lstm_model.tflite
# $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/dataset/*.bin \
# mnist_lstm_GENERATOR_INPUTS := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/dataset/*.bin)

mnist_lstm_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/lstm_model.cc

mnist_lstm_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/lstm_model.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lstm/*/Makefile.inc)

# TODO(b/268568089): This test is taking very long time to finish; causing the
# CI to run for a long time to finish.
ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifimini hifi3z))
  # Tests loading and running a vision model.
  $(eval $(call microlite_test,mnist_lstm_test,\
  $(mnist_lstm_TEST_SRCS),$(mnist_lstm_TEST_HDRS),$(mnist_lstm_GENERATOR_INPUTS)))
endif

# Tests the image provider module.
$(eval $(call microlite_test,image_provider_test,\
$(IMAGE_PROVIDER_TEST_SRCS),$(IMAGE_PROVIDER_TEST_HDRS)))

# Tests the detection responder module.
$(eval $(call microlite_test,detection_responder_test,\
$(DETECTION_RESPONDER_TEST_SRCS),$(DETECTION_RESPONDER_TEST_HDRS)))

# Builds a standalone object recognition binary.
$(eval $(call microlite_test,mnist_lstm,\
$(mnist_lstm_SRCS),$(mnist_lstm_HDRS),$(mnist_lstm_GENERATOR_INPUTS)))

# Add sources and headers generated from $(mnist_lstm_GENERATOR_INPUTS).
mnist_lstm_SRCS += $(mnist_lstm_GENERATED_SRCS)
mnist_lstm_HDRS += $(mnist_lstm_GENERATED_HDRS)

list_mnist_lstm_example_sources:
	@echo $(mnist_lstm_SRCS)

list_mnist_lstm_example_headers:
	@echo $(mnist_lstm_HDRS)
