import math

selection = -math.inf

while selection != -1:
    selection = input("\n1) Question 2.1 \n2) Question 2.2 \n3) Question 2.3 \n4) Question 3.1 \n5) Question 3.2 \n6) Quit \n\nYou must run 1 before 2 and 3.\nEnter your selection (1, 2, ..., 6): ")
    try:
        selection = int(selection)
    except:
        selection = -math.inf
    if selection == 1:
        print("\nQuestion 2.1 is running.\n------------------------")
        exec(open("question_2_1.py").read())
        print("Question 2.1 is done.\n")
    elif selection == 2:
        print("\nQuestion 2.2 is running.\n------------------------")
        exec(open("question_2_2.py").read())
        print("Question 2.2 is done.\n")
    elif selection == 3:
        print("\nQuestion 2.3 is running.\n------------------------")
        exec(open("question_2_3.py").read())
        print("Question 2.3 is done.\n")
    elif selection == 4:
        print("\nQuestion 3.1 is running.\n------------------------")
        exec(open("question_3_1.py").read())
        print("Question 3.1 is done.\n")
    elif selection == 5:
        print("\nQuestion 3.2 is running.\n------------------------")
        exec(open("question_3_2.py").read())
        print("Question 3.2 is done.\n")
    elif selection == 6:
        selection = -1 
    else:
        print("\nInvalid selection.")
        selection = -math.inf