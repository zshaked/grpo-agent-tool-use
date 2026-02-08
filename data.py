import random
import re

def generate_calculator_problems(n=20):
    """
    Generate problems that require calculator tool use.
    
    Examples:
    - "What is 47 times 83?"
    - "What is 15% of 847?"
    - "What is 23 squared?"
    """
    problems = []
    
    for _ in range(n):
        choice = random.randint(1, 4)
        
        if choice == 1:
            # Multiplication
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            question = f"What is {a} times {b}?"
            answer = str(a * b)
            
        elif choice == 2:
            # Percentage
            percent = random.randint(5, 50)
            number = random.randint(100, 999)
            question = f"What is {percent}% of {number}?"
            answer = str(int(number * percent / 100))
            
        elif choice == 3:
            # Square
            n = random.randint(10, 50)
            question = f"What is {n} squared?"
            answer = str(n ** 2)
            
        else:
            # Division
            b = random.randint(10, 50)
            a = b * random.randint(10, 99)  # Ensure clean division
            question = f"What is {a} divided by {b}?"
            answer = str(a // b)
        
        problems.append({
            "question": question,
            "answer": answer
        })
    
    return problems


if __name__ == "__main__":
    # Test data generation
    problems = generate_calculator_problems(5)
    for p in problems:
        print(f"Q: {p['question']}")
        print(f"A: {p['answer']}\n")