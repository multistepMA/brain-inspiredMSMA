
import numpy as np
import tensorflow as tf
tf.__version__

def MultiscaleTCNBlock(inputs, model_width, _drate):

    x0 = tf.keras.layers.Conv1D(model_width, 10, dilation_rate=_drate, padding='causal')(inputs)
    x1 = tf.keras.layers.Conv1D(int((model_width)/2), 10, dilation_rate=_drate, padding='causal')(x0)

    x0 = tf.keras.layers.BatchNormalization()(x0) 
    x0 = tf.keras.layers.Activation(tf.nn.relu)(x0) 

    x2 = tf.keras.layers.Conv1D(int((model_width)/4), 10, dilation_rate=_drate, padding='causal')(x1)

    x1 = tf.keras.layers.BatchNormalization()(x1) 
    x1 = tf.keras.layers.Activation(tf.nn.relu)(x1) 
    x2 = tf.keras.layers.LayerNormalization()(x2) 
    x2 = tf.keras.layers.Activation(tf.nn.relu)(x2) 
    _out = tf.keras.layers.concatenate([x0, x1, x2], axis = -1)
    _out = tf.keras.layers.Conv1D(model_width * _drate, 1, padding='same')(_out)
    _out = tf.keras.layers.BatchNormalization()(_out) 
    _out = tf.keras.layers.Activation(tf.nn.relu)(_out) 
    return _out 

def Add_block(input1, argv):
    # Concatenation Block from the KERAS Library
    cat = input1
    
    cat = tf.keras.layers.Add()([cat, argv])

    return cat

def Global_block(inputs, model_width, _drate):
    _global_output = MultiscaleTCNBlock(inputs, model_width, _drate)
    _global_output = tf.keras.layers.MaxPooling1D()(_global_output)
    _global_output = MultiscaleTCNBlock(_global_output, model_width, _drate)
    _global_output = tf.keras.layers.MaxPooling1D()(_global_output)
    _global_output = MultiscaleTCNBlock(_global_output, model_width, _drate)
    _global_output = tf.keras.layers.MaxPooling1D()(_global_output)

    return _global_output

def Global_block_propa(inputs, model_width, _drate):
    _global_output = MultiscaleTCNBlock(inputs, model_width, _drate)
    _global_output = MultiscaleTCNBlock(_global_output, model_width, _drate)
    _global_output = MultiscaleTCNBlock(_global_output, model_width, _drate)

    return _global_output


def GSPU(_input1, _input2, model_width, _drate, _prop):
    if _prop == True: 
        _Gb1 = Global_block_propa(_input1, model_width, _drate)
        _Gb2 = Global_block(_input2, model_width, _drate)

        _sim = tf.multiply(_Gb1, _Gb2) / (tf.norm(_Gb1) * tf.norm(_Gb2))
        _simGout = _sim * _Gb2
        _GBsum = tf.keras.layers.Add()([_Gb1, _Gb2])
        _simGout = tf.keras.layers.Add()([_GBsum, _simGout])

    else:
        _simGout = Global_block(_input1, model_width, _drate)
        _w_init = tf.random_normal_initializer()
        _sim = tf.Variable(initial_value = _w_init(shape = (_simGout.shape[1:]), dtype = 'float32'))
        _Gb1 = _simGout
        _Gb2 = []

    return _simGout, _Gb1, _Gb2


def act_fun(_spike_firing):
    _temp = tf.abs(_spike_firing) > 0.5
    
    return _spike_firing * tf.cast(_temp, tf.float32)

def _hebb_update(_alpha, _in, _spike, _mem, _hebb, model_width, _drate, _threshold):
    _decay = 0.5
    state = tf.keras.layers.Dense(model_width * _drate)(_in)
    state = tf.keras.layers.Add()([state, _alpha * (_in * tf.keras.layers.Dense(model_width * _drate)(_hebb))])
    _mem_poten = (_mem - _spike * _threshold) * _decay + state
    _spike_timing = act_fun(_mem_poten - _threshold)
    _hebb_info = 0.95 * _hebb
    _hebb_info = tf.clip_by_value(_hebb_info, -4, 4)

    return _mem_poten, _spike_timing, _hebb_info


def _LTP_change(G1, G2, _hebb_g, _spike_g, _mem_g1, _threshold, model_width, _drate, _prop):

    if _prop == True:
        _mem_g, _spike_GB1, _hebb_g1 = _hebb_update(_hebb_g, G1, _spike_g, _mem_g1, _hebb_g, model_width, _drate, _threshold)
    
        _mem_g, _spike_GB2, _hebb_g2 = _hebb_update(_hebb_g1, G2, _spike_GB1, _mem_g, _hebb_g1, model_width, _drate, _threshold)
        _mem_g = _mem_g / _threshold   

    else:
        _mem_g, _spike_GB2, _hebb_g2 = _hebb_update(_hebb_g, G1, _spike_g, _mem_g1, _hebb_g, model_width, _drate, _threshold)
        _mem_g = _mem_g / _threshold       
        
    return _spike_GB2, _mem_g, _hebb_g2 

def neural_sim_change_block(inp1, inp2, _hebb_g, _spike_g, _mem_g1, _threshold, model_width, _drate, prop_TF):
    _GSIn1toIn1, _GB1, GB2 = GSPU(inp1, inp2, model_width, _drate, prop_TF)

    mem_global1, _sp_global1, _h_global1 = _LTP_change(_GB1, GB2, _hebb_g, _spike_g, _mem_g1, _threshold, model_width, _drate, prop_TF)
 
    _GSIn1toIn1 = Add_block(_GSIn1toIn1, mem_global1)

    return mem_global1, _sp_global1, _h_global1, _GSIn1toIn1

class MSFMA_long:
    def __init__(self, length, num_channel, model_width, kernel_size, _drate, output_nums=1, _multi_forecast=True):
        # length: Input Signal Length
        # model_width: Width of the Input Layer of the Model
        # num_channel: Number of Channels allowed by the Model
        # kernel_size: Kernel or Filter Size of the Convolutional Layers
        # _drate: dilation rate the Convolutional Layers
        # output_nums: Output Classes 
        # _multi_forecast: True or False of Multi-step forecasting module 

        self.length = length
        self.num = num_channel
        self.model_width = model_width
        self.kernel_size = kernel_size
        self._drate = _drate
        self._multi_forecast = _multi_forecast
        self.output_nums = output_nums

    def HGSPU(self):
    

        input1 = tf.keras.Input((int(self.length/2), self.num))
        input2 = tf.keras.Input((int(self.length/2), self.num))
        input3 = tf.keras.Input((int(self.length/2), self.num))
        input4 = tf.keras.Input((int(self.length/2), self.num))
        input5 = tf.keras.Input((int(self.length/2), self.num)) 
        input6 = tf.keras.Input((int(self.length/2), self.num)) 

        _w_init = tf.random_normal_initializer()
        _hebb_g =  tf.Variable(initial_value = _w_init(shape = (int(self.length/16), self.model_width)), dtype = 'float32') 
        _spike_g = tf.Variable(initial_value = _w_init(shape = (int(self.length/16), self.model_width)), dtype = 'float32')
        _mem_g1 = tf.Variable(initial_value = _w_init(shape = (int(self.length/16), self.model_width)), dtype = 'float32')
        _threshold = 0.5
 
        mem_global1, _sp_global1, _h_global1, _GSIn1toIn1 = neural_sim_change_block(input1, [], [], _hebb_g, _spike_g, _mem_g1, _threshold, self.model_width, self._drate, False)

        mem_global2, _sp_global2, _h_global2, _GSIn1toIn2 = neural_sim_change_block(_GSIn1toIn1, input2, _h_global1, _sp_global1, mem_global1, _threshold, self.model_width, self._drate)        

        mem_global3, _sp_global3, _h_global3, _GSIn2toIn3 = neural_sim_change_block(_GSIn1toIn2, input3, _h_global2, _sp_global2, mem_global2, _threshold, self.model_width, self._drate, True)  

        mem_global4, _sp_global4, _h_global4,_GSIn3toIn4 = neural_sim_change_block(_GSIn2toIn3, input4, _h_global3, _sp_global3, mem_global3, _threshold, self.model_width, self._drate, True)  

        mem_global5, _sp_global5, _h_global5, _GSIn4toIn5 = neural_sim_change_block(_GSIn3toIn4, input5, _h_global4, _sp_global4, mem_global4, _threshold, self.model_width, self._drate, True)  

        _, _, _, _, _, _, _GSIn5toIn6 = neural_sim_change_block(_GSIn4toIn5, input6, _h_global5, _sp_global5, mem_global5, _threshold, self.model_width, self._drate, True)  


        outputs = _GSIn5toIn6

        if self._multi_forecast  == True:
            _outputs_gap_fir = tf.keras.layers.GlobalAveragePooling1D(data_format = 'channels_first')(outputs)
            _outputs_gap_fir = tf.keras.layers.Dense(187//16)(_outputs_gap_fir)
            _outputs_gap_fir = tf.keras.layers.Activation('relu')(_outputs_gap_fir)
            _outputs_gap_fir = tf.keras.layers.Dense(187)(_outputs_gap_fir)
            _outputs_gap_fir = tf.keras.layers.Activation('sigmoid')(_outputs_gap_fir)


            _outputs_gap_lst = tf.keras.layers.GlobalAveragePooling1D(data_format = 'channels_last')(outputs)
            _outputs_gap_lst = tf.keras.layers.Dense(62//16)(_outputs_gap_lst)
            _outputs_gap_lst = tf.keras.layers.Activation('relu')(_outputs_gap_lst)
            _outputs_gap_lst = tf.keras.layers.Dense(62)(_outputs_gap_lst)
            _outputs_gap_lst = tf.keras.layers.Activation('sigmoid')(_outputs_gap_lst)

            _outputs_concat = tf.keras.layers.Concatenate()([_outputs_gap_fir, _outputs_gap_lst])

            _outputs = tf.keras.layers.Dense(62, name='features')(_outputs_concat)
            _outputs = tf.keras.layers.Activation('relu')(_outputs)

        else:
            outputs = tf.keras.layers.Flatten()(outputs)
        _outputs = tf.keras.layers.Dense(32)(_outputs)
        _outputs = tf.keras.layers.Activation('relu')(_outputs)
        _outputs = tf.keras.layers.Dense(self.output_nums*2)(_outputs)
        _outputs = tf.keras.layers.Reshape([self.output_nums, 2])(_outputs)
        _outputs = tf.keras.layers.Softmax()(_outputs)


        model = tf.keras.Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=[_outputs])

        return model
