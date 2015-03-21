add_executable ( gen-random-bout gen-random-bout.cpp )

add_dependencies ( gen-random-bout LLVM )

target_link_libraries( gen-random-bout kleeBasic )