




def main(): 
    a = [1,2,3,4,5]
    b = [6,7,8,9,10]
    c = [11, 12, 13, 14, 15]

    my_list = [a,b,c]
    print(list(zip(*my_list)))

if __name__ == "__main__": 
    main()