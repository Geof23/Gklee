declare void @print_i32(i32)

define i32 @main() {
entry:
	br label %test
test:
	%a = phi i32 [10, %entry], [%b, %test]
	%b = phi i32 [%a, %test], [20, %entry]
	%c = phi i32 [0, %entry], [1, %test]
	%d = icmp eq i32 %c, 1
	br i1 %d, label %exit, label %test
exit:
	call void @print_i32(i32 %b)
	ret i32 0
}
