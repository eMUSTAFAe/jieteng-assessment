import math
def softmax(xs):
    m=max(xs); ex=[math.exp(x-m) for x in xs]; s=sum(ex); return [x/s for x in ex]
def attention(q,k,v,causal=False):
    output=[]
    for i,query in enumerate(q):
        scores=[sum(a*b for a,b in zip(query,key))/math.sqrt(len(query)) if not causal or j<=i else -1e9 for j,key in enumerate(k)]
        weights=softmax(scores); output.append([sum(w*value[d] for w,value in zip(weights,v)) for d in range(len(v[0]))])
    return output
if __name__ == '__main__':
    print(attention([[1.,0.],[0.,1.]],[[1.,0.],[0.,1.]],[[1.,2.],[3.,4.]],True))
