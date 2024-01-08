from k_means import k_means, image_compression


if __name__ == "__main__":

    #image_name = 'colors.png' or 'mountain.png' or 'bridge.png'
    #initialization = 'random' or 'k-means++'
    iterations = 15
    initialization = 'random'
    
    
    print("The results of the tests will be stored in the directory 'results'.\n")
    #----------------------------------------
    #Test1:
    print('Test 1')
    image_compression('mountain.png',32,iterations,'random')
    print('Test 1 completed.')
    
    #Test2:
    print('\nTest 2')
    for k in [8,16,32]:
        print(f'k = {k}')  
        image_compression('bridge.png',k,iterations,initialization)
    print('\nTest 2 completed.')   
    
    #Test3:
    print('\nTest 3')
    for init in ['random','k-means++']:
        print(f'\nInitialization: {init}')
        image_compression('colors.png',7,iterations,init)
    print('Test 3 completed.')
    
    print("Tests completed. Please check the folder 'results'.")