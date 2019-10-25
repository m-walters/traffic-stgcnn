#%source bin/activate
import tensorflow as tf
import random
import numpy as np
import time
import sys, getopt
from tensorflow.contrib import rnn

global_tstart = time.time()

def stdout(s):
    sys.stdout.write(str(s)+'\n')

nrod = 400
batchsize = -1
seq_len = -1 
nEpoch = -1
eta = 1e-2
nInput = nrod
nHidden = 32
nDense = 32
subnlayer = 1
seqnlayer = 1
bThetas = False

bSummaries = False
now = time.localtime()
fnow = time.strftime("%Y%m%d-%H%M%S",now) 
outftag = "seq_rnn_"+fnow
trnfile = ""
trnrange = None
testfile = ""
ckptfile = None
bSaveCkpt = False
stepsize = 1

arglen = len(sys.argv[1:])
arglist = sys.argv[1:]

try:
    opts, args = getopt.getopt(arglist,"s:e:b:o:",["trnfile=","trnfile_range=","testfile=",\
        "output=","bSummaries=","ckpt=","bSaveCkpt=","stepsize="])
except:
    stdout("Error in opt retrival...")
    stdout("seq_rnn")
    stdout("  -e num epoch")
    stdout("  -s sequence length")
    stdout("  -b batch size")
    stdout("  -o,--output output file tag")
    stdout("  --bSummaries record summaries (boolean)")
    stdout("  --ckpt checkpoint file")
    stdout("  --bSaveCkpt (boolean)")
    sys.exit(2)

for opt, arg in opts:
    if opt == "-e":
        nEpoch = int(arg)
    elif opt == "-s":
        seq_len = int(arg)
    elif opt == "-b":
        batchsize = int(arg)
    elif opt == "--trnfile":
        trnfile = arg
    elif opt == "--trnfile_range":
        trnrange = arg.split("-")
    elif opt == "--testfile":
        testfile = arg
    elif opt in ("-o","--output"):
        outftag = arg
    elif opt in ("--stepsize"):
        stepsize = int(arg)
    elif opt == "--bSummaries":
        if arg in ("False","false","0"):
            bSummaries = False
        elif arg in ("True","true","1"):
            bSummaries = True
        else:
            stdout("Fromat bSummaries properly")
            sys.exit(2)
    elif opt == "--bSaveCkpt":
        if arg in ("False","false","0"):
            bSaveCkpt = False
        elif arg in ("True","true","1"):
            bSaveCkpt = True
        else:
            stdout("Fromat bSaveCkpt properly")
            sys.exit(2)
    elif opt == "--ckpt":
        ckptfile = arg

summdir = "/home/walterms/project/walterms/mcmd/nn/tfrnn/summaries/"
ckptdir = "/home/walterms/project/walterms/mcmd/nn/tfrnn/ckpts/seq/"
outdir = "/home/walterms/project/walterms/mcmd/nn/data/scratch/xmelt/rnn_out/"
lossname = outdir+outftag+"_losses"

features = ["x","y","th"]
# features = ["x","y","ft1","ft2"]
featdict = {}
for ft in features:
    featdict.update({ft:[]})

nchannel = len(features)

def gen_seq_set(f,stepsize=1,nblMax=-1):

    stdout("Creating sequence array from "+f)
    '''
    # Count blocks
    fin = open(f,'r')
    nbl = 0
    for line in fin.readlines():
        if line == "\n":
            nbl+=1
    stdout("Num blocks in "+f+" "+str(nbl))
    stdout("Return size "+str(nbl//stepsize)+" (stepsize="+str(stepsize)+")")
    fin.close()
    '''

    sortIdx = np.arange(nrod,dtype=int)
    IDs = []
    fin = open(f, 'r')
    nbl = 0
    seqset = []

    writestep = 1
    for line in fin.readlines():
        if writestep != stepsize:
            if line == "\n":
                writestep += 1
            continue
        if line == "\n":
            # Done a block
            # Sort based on rod indices
            sortIdx = np.argsort(IDs)
            
            # Insert data as triplets
            channels = []
            for ft in features:
                channels.append(featdict[ft])
            prep_data = []
            for ch in channels:
                prep_data.append(np.asarray(ch)[sortIdx])
            formatted_data = np.stack(prep_data)
            seqset.append(formatted_data)
                
            for ft in features:
                featdict[ft] = []
            IDs = []
            nbl+=1
            writestep = 1
            if nbl == nblMax:
                break
            continue
        spt = [float(x) for x in line.split()]
        featdict["x"].append(spt[0])
        featdict["y"].append(spt[1])
        featdict["th"].append(spt[2])
        
        IDs.append(int(spt[3]))

    fin.close()

    return np.asarray(seqset)
    
###################
#       RNN       # 
###################

stdout("Creating RNN Graph")

def variable_summaries(var):
    #A ttach a lot of summaries to a Tensor (for TensorBoard visualization)
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


X = tf.placeholder("float", [None, seq_len, nchannel, nInput],name="X")
Y = tf.placeholder("float", [None, nchannel, nInput],name="Y")

with tf.name_scope('dense'):
    dense_weights = {"pre":tf.Variable(tf.random_normal([nHidden,nDense],
                stddev=0.1,dtype=tf.float32),name="pre_w")}
    for f in features:
        dense_weights.update({f:tf.Variable(tf.random_normal([nDense,nrod],
                stddev=0.1,dtype=tf.float32),name=f+"_w")})

    dense_biases = {"pre":tf.Variable(tf.random_normal([nDense],
                stddev=0.1,dtype=tf.float32),name="pre_b")}
    for f in features:
        dense_biases.update({f:tf.Variable(tf.random_normal([nrod],
                stddev=0.1,dtype=tf.float32),name=f+"_b")})
        
    for w in dense_weights:
        tf.summary.histogram(w+"_ws",dense_weights[w])
    for b in dense_biases:
        tf.summary.histogram(b+"_bs",dense_biases[b])


# Define an lstm cell with tensorflow
def lstm_cell(nUnits):
    return rnn.BasicLSTMCell(nUnits)

def seqRNN(x):

    x = tf.unstack(x,seq_len,1) # unstack along time dimension
    
    with tf.name_scope('subrnn'):
        with tf.variable_scope('subrnn'):
            # Subcell    
#             subcell = lstm_cell(nHidden)
            subcell = rnn.MultiRNNCell([lstm_cell(nHidden) for _ in range(subnlayer)])

            suboutputs = []
            substate = subcell.zero_state(batchsize,tf.float32)

            # Loop over the images in a sequence
            for x_img in x:
                x_ = tf.unstack(x_img,nchannel,1)
                # Returns multiple outputs I think of size [batchsize,nchannel,subcell.output_size]
                suboutput_img, substate = tf.nn.static_rnn(subcell,x_,dtype=tf.float32,initial_state=substate)
                # suboutput_img is a list of 3 outputs from each iteration on the img
                # suboutput_img[-1] is the last output, let's use that as input to the seqrnn
                suboutputs.append(suboutput_img[-1])

            tf.summary.histogram('substate',substate)

    with tf.name_scope('seqrnn'):
        with tf.variable_scope('seqrnn'):
            # Main cell
#             cell = lstm_cell(nHidden)
            cell = rnn.MultiRNNCell([lstm_cell(nHidden) for _ in range(seqnlayer)])

            outputs,state = tf.nn.static_rnn(cell,suboutputs,dtype=tf.float32)
            tf.summary.histogram('cellstate',state)


    # Dense output from seqrnn
    with tf.name_scope('dense'):
        dense_pre = tf.nn.elu(tf.add(tf.matmul(outputs[-1],dense_weights["pre"]),
                        dense_biases["pre"]),name="pre_out_activ")

        # Tensors for transforming output of main RNN unit into an img
        out_img_channels = []
        i = 0
        for ft in features:
            out_img_channels.append(tf.nn.tanh(tf.add(tf.matmul(
                dense_pre,dense_weights[ft]),dense_biases[ft]),name=str(ft)+"_out_activ"))

            tf.summary.histogram(str(ft)+"_out",out_img_channels[-1])
            i+=1
    
    return tf.stack(out_img_channels,axis=1)


# Outputs a list of tensors of size nrod representing the img
seq_img = seqRNN(X)

# Define loss and optimizer
loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=Y, predictions=seq_img))
tf.summary.scalar('loss',loss)

optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)

stdout("Finished Graph")


###################
#    TRAINING     # 
###################

# Saver for checkpoints
saver = tf.train.Saver()

epochEval = int(10**(np.log10(nEpoch)//1 - 1))
if epochEval<1: epochEval=1

outlosses = []
floss = open(lossname, 'w')

# Generate training list
trnlist = [trnfile]
if trnrange:
    trnlist = []
    first, last = int(trnrange[0]),int(trnrange[1])
    for i in range(last-first+1):
        trnlist.append(trnfile+"_"+str(i))

stdout("Generating testing data...")
test_seq = gen_seq_set(testfile,stepsize=stepsize)
stdout("Done")
nTestSeq = len(test_seq)-seq_len
nTestSample = (600//batchsize + 1)*batchsize
ntestbatches = nTestSample//batchsize
imgIdx_test = [i for i in range(nTestSeq)]
nextra_test = ntestbatches*batchsize - nTestSeq
stdout(str(len(test_seq))+" images in test set")
stdout(str(batchsize*ntestbatches)+" sequences per epoch")

stdout("Beginning Session")
with tf.Session() as sess:
    if bSummaries:
        summaries = tf.summary.merge_all()
        now = time.localtime()
        writeto = summdir+time.strftime("%Y%m%d-%H%M%S",now) + "/"
        train_writer = tf.summary.FileWriter(writeto+"train", sess.graph)
        test_writer = tf.summary.FileWriter(writeto+"test")

    # Checkpoint file
    if ckptfile:
        ckptfile = ckptdir+ckptfile
        stdout("Restoring from "+ckptfile)
        saver.restore(sess, ckptfile)
        stdout("Model restored")
    else:
        stdout("No checkpoint file given for model restore")
        ckptfile = ckptdir+"default_s"+str(seq_len)+"_b"+str(batchsize)+"_ss"+str(stepsize)+".ckpt"
        stdout("Initializing variables")
        sess.run(tf.global_variables_initializer())

    tstart = time.time()
    trnstep = 0

    # Train over the training list
    for trnf in trnlist:
        # Generate seq sets
        stdout("Generating seq from "+trnf+"...")
        try:
            trn_seq = gen_seq_set(trnf,stepsize=stepsize)
        except:
            stdout("Failed to generate sequence, trying next file")
            continue
        stdout("Done")

        nTrnSeq = len(trn_seq)-seq_len

        # Add +1 to batches per
        batchesPerEpoch = nTrnSeq//batchsize

        imgIdx_trn = [i for i in range(nTrnSeq)]

        stdout(str(len(trn_seq))+" images in train set")
        stdout(str(batchesPerEpoch*batchsize)+" sequences per epoch")

        metarecord_ib = int(10**(np.log10(batchesPerEpoch)//1 - 1))
        if metarecord_ib<1: metarecord_ib=1
    
        for e in range(nEpoch):
            trn_loss = 0.
            random.shuffle(imgIdx_test)
            random.shuffle(imgIdx_trn)
            start = 0
            for ib in range(batchesPerEpoch):
                trnstep += 1
                # Pepare a batch
                end = start+batchsize
                yin = np.asarray([trn_seq[i_img+seq_len] \
                        for i_img in imgIdx_trn[start:end]])
                xin = np.asarray([[trn_seq[i_img+s] for s in range(seq_len)] \
                        for i_img in imgIdx_trn[start:end]])
                
                start = end
                
                if ib%metarecord_ib==0 and bSummaries:
                    # Record run metadata
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    _,l,S = sess.run([optimizer, loss, summaries], feed_dict={X: xin, Y: yin},
                                    options=run_options, run_metadata=run_metadata)

                    train_writer.add_run_metadata(run_metadata, "step %d"%(trnstep))
                    train_writer.add_summary(S, trnstep)
                    
                else:
                    _,l = sess.run([optimizer,loss], feed_dict={X:xin,Y:yin})

                trn_loss += l / batchesPerEpoch
                
            if e % epochEval == 0:
                # Eval on test set
                test_loss = 0.
                start = 0
                for tb in range(ntestbatches):
                    end = start+batchsize
                    yin = np.asarray([test_seq[i_img+seq_len] \
                            for i_img in imgIdx_test[start:end]])
                    xin = np.asarray([[test_seq[i_img+s] for s in range(seq_len)] \
                            for i_img in imgIdx_test[start:end]])
                    l, = sess.run([loss],feed_dict={X:xin,Y:yin})
                    test_loss += l/ntestbatches

                    start = end

                tend = time.time()
                stdout("(t"+str(trnstep)+") epoch "+str(e)+"  trn_loss "+'%.6f'%(trn_loss)+\
                    "  test_loss "+'%.6f'%(test_loss)+"  elapsed time(s) "+str((tend-tstart)))
                outlosses.append((trnstep,trn_loss))
                tstart = time.time()
        
    stdout("Done Training")

    
    if bSaveCkpt:
        # Saving checkpoint
        stdout("Saving checkpoint to "+ckptfile)
        save_path = saver.save(sess, ckptfile)
        stdout("Saved checkpoint")

    if bSummaries:
        train_writer.close()
        test_writer.close()

    
floss.write("nEpoch %d | batchsize %d | nTrn %d | nTest %d | SeqLen %d | eta %.5f | nHidden %d | nDense %d | subnlayer %d | seqnlayer %d\n"%(nEpoch,batchsize,len(trn_seq),len(test_seq),seq_len,eta,nHidden,nDense,subnlayer,seqnlayer))

floss.write("data header: trnstep trn_loss")
for t,l in outlosses:
    floss.write("%d %f\n"%(int(t),l))
sess.close()

global_tend = time.time()
stdout("Total time: "+str(global_tend-global_tstart))
