

def main(): 
    my_dict = {str(i) : i for i in range(10)}

    for key in my_dict: 
        print(key,":", my_dict[key])

if __name__ == "__main__": 
    main()