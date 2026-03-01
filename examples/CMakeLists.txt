cmake_minimum_required(VERSION 3.22)

add_executable(run_inference run_inference.cpp)
target_link_libraries(run_inference PRIVATE servoflow)
target_include_directories(run_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_executable(rdt1b_inference rdt1b_inference.cpp)
target_link_libraries(rdt1b_inference PRIVATE servoflow)
target_include_directories(rdt1b_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include)
