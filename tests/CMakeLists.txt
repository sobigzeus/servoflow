cmake_minimum_required(VERSION 3.22)

# ── Unit tests ────────────────────────────────────────────────────────────────
function(sf_add_test name)
    add_executable(${name} unit/${name}.cpp)
    target_link_libraries(${name} PRIVATE servoflow GTest::gtest_main)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )
    gtest_discover_tests(${name})
endfunction()

include(GoogleTest)

sf_add_test(test_dtype)
sf_add_test(test_shape)
sf_add_test(test_tensor)
sf_add_test(test_sampler)
sf_add_test(test_quantization)

# Integration tests require CUDA.
if(SF_ENABLE_CUDA)
    add_executable(test_vla_pipeline integration/test_vla_pipeline.cpp)
    target_link_libraries(test_vla_pipeline PRIVATE servoflow GTest::gtest_main)
    gtest_discover_tests(test_vla_pipeline)

    # Alignment test: loads real RDT-1B checkpoint (skipped if not present).
    add_executable(test_rdt1b_load alignment/test_rdt1b_load.cpp)
    target_link_libraries(test_rdt1b_load PRIVATE servoflow GTest::gtest_main)
    gtest_discover_tests(test_rdt1b_load)

    # E2E alignment binary: runs forward_raw and writes output for comparison.
    add_executable(align_rdt1b alignment/align_rdt1b.cpp)
    target_link_libraries(align_rdt1b PRIVATE servoflow)

    # Kernel benchmark: measures pure forward_raw latency with CUDA events.
    add_executable(bench_rdt1b_kernel alignment/bench_rdt1b_kernel.cpp)
    target_link_libraries(bench_rdt1b_kernel PRIVATE servoflow)
endif()
