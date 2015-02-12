 #!/bin/bash
find . -type f -not -name "*.cu" -not -name "*.sh" -not -name "*.py" -not -name "expected.txt" -not -name "README" -not -name "options.txt" | xargs rm -fv
rm -vrf */klee*