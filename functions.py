
# returns True if the argument passed is even

def check_even(number):
    if number % 2 == 0:
          return True  

    return False


numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# if an element passed to check_even() returns True, select it
even_numbers_iterator = filter(check_even, numbers)


# converting to list
even_numbers = list(even_numbers_iterator)

print(even_numbers)

#fonction lambda
x = lambda a : a + 10
print(x(5))

#fonction map()
# appliquer une fonction sur un itérateur

def myfunc(n):
  return len(n)

x = map(myfunc, ('apple', 'banana', 'cherry'))