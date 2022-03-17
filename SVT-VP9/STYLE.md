This style guide comes from `dav1d`: https://code.videolan.org/videolan/dav1d/wikis/Coding-style

no tabs, only spaces;
4-space indentation;
for multi-line statements, the indentation of the next line depends on the context of the statement and braces around it. For example, if you have a long assignment, you can choose to either align it to the = of the first line, or (if that leads to less lines of code) just indent 1 level further from the first line's indentation level:

```
const int my_var = something1 &&
                   something2;
```
or
```
const int my_var = something1 +
    something2 - something3 * something4;
```
However, if there are braces, the first non-whitespace character of the line should be aligned with the brace level that it is part of:
```
const int my_var = (something1 +
                    something2) * something3;
```

use `CamelCase` for types and `under_score` for variable names (`TypeName my_instance;`)
we use const where possible, except in forward function declarations in header files, where we only use it for const-arrays:

`int my_func(const array *values, int arg);`

[..]

```
int my_func(const array *const values, const int num) {
    ..
}
```

braces go on the same line for single-line statements, but on a new line for multi-line statements:

```
static void function(const int argument) {
    do_something();
}
```
versus
```
static void function(const int argument1,
                     const int argument2)
{
    do_something();
}
```

braces are only necessary for multi-line code blocks or multi-line condition statements;

```
if (condition1 && condition2)
    do_something();
```
and
```
if (condition) {
    do_something_1();
    do_something_2();
}
```
and
```
if (condition1 &&
    condition2)
{
    do_something();
}
```

switch/case are indented at the same level, and the code block is indented one level deeper:

```
switch (a) {
case 1:
    bla();
    break;
}
```
but for very trivial blocks, you can also put everything on one single line
```
switch (a) {
case 1: bla(); break;
}
```

lines should not be longer than 80 characters. We allow exceptions if wrapping the line would lead to exceptional ugliness, and this is done on a case-by-case basis;
don't use goto except for standard error handling;
use native types (`int`, `unsigned`, etc.) for scalar variables where the upper bound of a size doesn't matter;
use sized types (`uint8_t`, `int16_t`, etc.) for vector/array variables where the upper bound of the size matters;
use dynamic types (`pixel`, `coef`, etc.) so multi-bitdepth templating works as it should.
