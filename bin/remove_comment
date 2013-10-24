#!/usr/bin/perl -w

$/ = undef;   # no line delimiter
$_ = <>;   # read entire file

s! ((['"]) (?: \\. | .)*? \2) | # skip quoted strings
   /\* .*? \*/ |  # delete C comments
   // [^\n\r]*   # delete C++ comments
 ! $1 || ' '   # change comments to a single space
 !xseg;    # ignore white space, treat as single line
    # evaluate result, repeat globally
print;
