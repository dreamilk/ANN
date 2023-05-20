cmake_minimum_required(VERSION 3.22)
project(ann)

include_directories(./include)

add_definitions("-g -w")

aux_source_directory(./src SRC_LIST)

add_executable(ann ${SRC_LIST})

install(TARGETS ann RUNTIME DESTINATION ./)

