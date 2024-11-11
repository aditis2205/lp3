def fibo_nonrecursive(n):
    n1,n2=0,1
    count=0
    if n<=0:
        print("Please enter a positive integer")
        return -1
    elif n == 1:
        print(n1)
    elif n == 2:
        print(n1, n2)
    else:
        while(count<n):
            print(n1,end=" ")
            nth=n1+n2
            n1=n2
            n2=nth
            count+=1
            
def Fibonacci(n, n1=0, n2=1):
    if n <= 0:
        print("Incorrect input")
        return
    elif n == 1:
        print(n1, end=" ")  
    else:
        print(n1, end=" ")  
        Fibonacci(n - 1, n2, n1 + n2)  
    
    
if __name__ == '__main__':
    n=int(input("Enter the number of terms:"))
    print("Fibonacci series upto" ,n, "term is:")
    fibo_nonrecursive(n)
    print(Fibonacci(n))