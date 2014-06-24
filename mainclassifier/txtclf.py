import nltk
import pickle

aneglist = pickle.load( open( "aneglist.pickle", "rb" ) )
aposlist = pickle.load( open( "aposlist.pickle", "rb" ) )
nneglist = pickle.load( open( "nneglist.pickle", "rb" ) )
nposlist = pickle.load( open( "nposlist.pickle", "rb" ) )
rneglist = pickle.load( open( "rneglist.pickle", "rb" ) )
rposlist = pickle.load( open( "rposlist.pickle", "rb" ) )
vneglist = pickle.load( open( "vneglist.pickle", "rb" ) )
vposlist = pickle.load( open( "vposlist.pickle", "rb" ) )
dicy = pickle.load( open( "dicy.pickle", "rb" ) )


def valu(c,t):
    val=(0,0)
    tt=c
    if c in dicy:
            if ('JJ' in t):
                val=(aposlist["%s" % tt],aneglist["%s" % tt])
            elif ('NN' in t):
                val=(nposlist["%s" % tt],nneglist["%s" % tt])
            elif ('RB' in t):
                return 0
            elif ('VB' in t):
                val=(vposlist["%s" % tt],vneglist["%s" % tt])
    return val

while(1):
    st=raw_input("GIMME INPUT : ")
    wds=st.split()
    post=nltk.pos_tag(wds)
    psum=0
    nsum=0
    a=0
    b=0
    fg=0
    for i in range(0,len(wds)):
        t=post[i][1]
        tt=(post[i][0].lower())
        if tt in dicy and fg==1:
            tup=valu(tt,t)
            if(tup!=0):
                if (a!=b):
                    psum+=(2*(a-b)*tup[0])
                    nsum+=(2*(a-b)*tup[1])
                    fg=0
                else:
                    psum+=(tup[0])
                    nsum+=(tup[1])
            else:
                a+=rposlist["%s" % tt]
                b+=rneglist["%s" % tt]
        elif tt in dicy:
            tup=valu(tt,t)
            if(tup!=0):
                psum+=(tup[0])
                nsum+=(tup[1])
            else:
                a=rposlist["%s" % tt]
                b=rneglist["%s" % tt]
                fg=1

    tot=psum+nsum 
    print "On a scale of -1 to 1, these are the scores"
    print "Positive score : ",psum
    print "Negative score : ",nsum,"\n"        

