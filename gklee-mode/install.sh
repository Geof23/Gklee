#!/bin/bash
mkdir ~/.emacs.d/el_files &> /dev/null
cp *.el ~/.emacs.d/el_files/


# from http://www.computing.net/answers/programming/oneliner-to-insert-text-if-found/17638.html
# look for (add-to-list (quote load-path) "~/.emacs.d/el_files") in .emacs
# if not there add it to the end
#perl -pi.bak -e 'BEGIN{$str = "(add-to-list (quote load-path) \"~/.emacs.d/el_files\")";}$seen++ if /$str/; END{print $str if !$seen}' ~/.emacs

# same with (load "gklee-compile")
#perl -pi.bak -e 'BEGIN{$str = "(load \"gklee-compile\")";}$seen++ if /$str/; END{print $str if !$seen}' file.txt'}""}'
