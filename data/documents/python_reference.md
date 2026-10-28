# Python Functions

## What Are Python Functions?

A function in Python is a reusable block of code designed to perform a specific task. Functions help organize programs into smaller, manageable parts, reduce repeated code, and improve readability.

A function is defined using the `def` keyword, followed by its name, parentheses, and a colon. The function body contains the statements to execute when the function is called. A function can accept inputs called parameters and can return a result using the `return` statement.

## How Are Python Functions Used?

Functions are called by writing their name followed by parentheses. Arguments can be passed into the parentheses to provide input values.

For example:

```python
def add(a, b):
    return a + b

result = add(3, 5)
print(result)  # 8
```

In this example, `add` is the function name, `a` and `b` are parameters, `3` and `5` are arguments, and `return` sends the calculated result back to the caller.

## Common Types of Functions

- **Built-in functions:** Provided by Python, such as `print()`, `len()`, and `sum()`.
- **User-defined functions:** Created by programmers using `def`.
- **Lambda functions:** Small anonymous functions created using the `lambda` keyword.

Functions improve code reuse, modularity, testing, and maintainability.