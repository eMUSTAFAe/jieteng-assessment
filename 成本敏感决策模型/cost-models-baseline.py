ROWS=[(0.1,0),(0.4,1),(0.6,0),(0.9,1)]
def evaluate(threshold, fp=1, fn=4):
    errors=[fp if p>=threshold and y==0 else fn if p<threshold and y==1 else 0 for p,y in ROWS]
    return sum(errors)/len(errors)
if __name__ == '__main__':
    print({t:evaluate(t) for t in [0.2,0.5,0.8]})
