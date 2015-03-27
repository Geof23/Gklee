message( "let's see if we can understand how lists can be defined and used in an external cmake script call (-P) so that we can iterate the items separately or use them collectively as needed . . ." )

file( GLOB ITEMS * )

message( "ITEMS in Test.cmake is ${ITEMS}" )
message( "or: " ${ITEMS} )

message( "here is the command line: COMMAND ${CMAKE_COMMAND} -DITEMS=${ITEMS} -P  ${CMAKE_CURRENT_LIST_DIR}/Test2.cmake" )

execute_process( COMMAND ${CMAKE_COMMAND} "-DNITEMS=${ITEMS}" -P  ${CMAKE_CURRENT_LIST_DIR}/Test2.cmake )

message( "finished in Test.cmake" )
