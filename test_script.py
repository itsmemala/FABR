import sys,os,argparse,time

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(float, arg.split(',')))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_lamb', type=list_of_ints, default='')
    args = parser.parse_args()
    
    print(args.custom_lamb)
    
    return
    
if __name__ == '__main__':
    sys.exit(main())