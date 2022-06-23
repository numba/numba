#!/bin/bash

sed 's/0x[0-9a-f]*/HEX/g' annotation_without_tid.txt > nohex_annotation_without_tid.txt
sed 's/0x[0-9a-f]*/HEX/g' annotation_with_tid.txt > nohex_annotation_with_tid.txt
diff -u nohex_annotation_without_tid.txt nohex_annotation_with_tid.txt > tid_typing_change.diff
grep "^[+-]" tid_typing_change.diff > changes_only.diff
