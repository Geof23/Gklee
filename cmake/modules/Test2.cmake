message( "entered Test2 -- we're going to look at the NITEMS variable and try to iterate over it -- NITEMS is ${NITEMS}" )

foreach( I ${NITEMS} )
  message( "individual item is ${I}" )
endforeach(I)
