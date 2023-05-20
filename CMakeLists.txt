project(main)

include_directories(./include)

add_definitions("-g -w")

aux_source_directory(./src SRC_LIST)

add_executable(main ${SRC_LIST})


