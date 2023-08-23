mnist_lenet_MODEL_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/lenet_model_settings.cc

mnist_lenet_MODEL_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/lenet_model_settings.h

mnist_lenet_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/lenet_test.cc \
$(mnist_lenet_MODEL_SRCS)

mnist_lenet_TEST_HDRS := \
$(mnist_lenet_MODEL_HDRS)

DETECTION_RESPONDER_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/detection_responder_test.cc

DETECTION_RESPONDER_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/detection_responder.h

mnist_lenet_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/detection_responder.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/image_provider.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/main.cc \
$(mnist_lenet_MODEL_SRCS)

mnist_lenet_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/detection_responder.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/image_provider.h \
$(mnist_lenet_MODEL_HDRS)

mnist_lenet_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/lenet_model.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/dataset/*.bin \
# mnist_lenet_GENERATOR_INPUTS := $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/dataset/*.bin)

mnist_lenet_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/lenet_model.cc

mnist_lenet_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/models/lenet_model.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mnist_lenet/*/Makefile.inc)

# TODO(b/268568089): This test is taking very long time to finish; causing the
# CI to run for a long time to finish.
ifneq ($(TARGET_ARCH), $(filter $(TARGET_ARCH), hifimini hifi3z))
  # Tests loading and running a vision model.
  $(eval $(call microlite_test,mnist_lenet_test,\
  $(mnist_lenet_TEST_SRCS),$(mnist_lenet_TEST_HDRS),$(mnist_lenet_GENERATOR_INPUTS)))
endif

# Tests the image provider module.
$(eval $(call microlite_test,image_provider_test,\
$(IMAGE_PROVIDER_TEST_SRCS),$(IMAGE_PROVIDER_TEST_HDRS)))

# Tests the detection responder module.
$(eval $(call microlite_test,detection_responder_test,\
$(DETECTION_RESPONDER_TEST_SRCS),$(DETECTION_RESPONDER_TEST_HDRS)))

# Builds a standalone object recognition binary.
$(eval $(call microlite_test,mnist_lenet,\
$(mnist_lenet_SRCS),$(mnist_lenet_HDRS),$(mnist_lenet_GENERATOR_INPUTS)))

# Add sources and headers generated from $(mnist_lenet_GENERATOR_INPUTS).
mnist_lenet_SRCS += $(mnist_lenet_GENERATED_SRCS)
mnist_lenet_HDRS += $(mnist_lenet_GENERATED_HDRS)

list_mnist_lenet_example_sources:
	@echo $(mnist_lenet_SRCS)

list_mnist_lenet_example_headers:
	@echo $(mnist_lenet_HDRS)
