def print_quadratic_equation(a, b, c):
    equation = f"{a}x^2 + {b}x + {c} = 0"
    print("Quadratic Equation:", equation)

a = int(input("Enter the coefficient a: "))
b = int(input("Enter the coefficient b: "))
c = int(input("Enter the coefficient c: "))

print_quadratic_equation(a,b,c)
