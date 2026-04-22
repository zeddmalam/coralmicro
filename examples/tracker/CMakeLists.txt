add_executable_m7(tracker
    tracker.cc
    DATA
    ${PROJECT_SOURCE_DIR}/models/posenet_mobilenet_v1_075_324_324_16_quant_decoder_edgetpu.tflite
    ${PROJECT_SOURCE_DIR}/models/posenet_test_input_324.bin
)

target_link_libraries(tracker
    libs_base-m7_freertos
    libs_jpeg_m7
)
