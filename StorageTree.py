import os
import platform
from numpy import *
from utilFunctions import *

import persistent
import persistent.list

import transaction
from BTrees._OOBTree import OOBTree

def open_StorageTree_DB(path:str='',name:str='StorageTreeDB',new:bool=False):

    if new == True:
        import shutil
        from os import mkdir
        shutil.rmtree(path)
        mkdir(path)

    import ZODB
    import ZODB.FileStorage

    if not os.path.exists(path):
        os.mkdir(path)

    pfo = PathFileObj(root=path,file=name)

    storage = ZODB.FileStorage.FileStorage(pfo.filepath)
    db = ZODB.DB(storage)

    conn = db.open()

    try:
        stt = conn.root.stt
    except AttributeError:
        conn.root.stt = StorageTree(key=name)
        stt = conn.root.stt

    return stt

class StorageTree(persistent.Persistent):

    def __init__(self,key:str='head',parent=None) -> None:
        self._parent = parent
        self._key = str(key)
        self.node_storage = OOBTree()
        self.attribute_storage = OOBTree()

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self,key:str):
        old_key = self._key
        self._key = str(key)
        if not isinstance(self.parent,type(None)):
            self.parent.node_storage[self._key] = self.parent.node_storage.pop(old_key)
        transaction.commit()

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self,parent):
        self._parent = parent
        transaction.commit()

    def ga(self,attribute_key:str,attribute:str=None):

        if str(attribute_key) not in self.attribute_storage:
            self.attribute_storage.update({str(attribute_key):attribute})
        elif not isinstance(attribute,type(None)):
            self.attribute_storage.update({str(attribute_key):attribute})

        transaction.commit()
        return self.attribute_storage[str(attribute_key)]

    def gna(self,node_keys:list[str],attribute_key:str,attribute:str=None):

        cur_node = self.gns(node_keys)

        return cur_node.ga(attribute_key,attribute)

    def gn(self,node_key:str):

        if str(node_key) not in self.node_storage:
            self.node_storage.update({str(node_key):StorageTree(key=node_key,parent=self)})

        transaction.commit()
        return self.node_storage[str(node_key)]

    def gns(self,node_keys:list[str]):

        cur_node = self

        for key in node_keys:
            cur_node = cur_node.gn(key)

        return cur_node

    def gps(self):

        if not isinstance(self.parent,type(None)):
            parents = self.parent.gps()
            parents.append(self.key)
        else:
            return list()

        return parents

    def get_nodes(self,node_keys:list[str],including:bool=True):

        if including == True:
            return [self.gn(node_key) for node_key in node_keys]
        
        elif including == False:
            return [self.gn(node_key) for node_key in list(self.all_node_keys()) 
                    if node_key not in node_keys]
    
    def add_tree(self,tree,copy:bool=False):

        if copy == True:
            tree = tree.copy_tree()

        self.node_storage.update({tree.key:tree})
        tree.parent = self

        transaction.commit()

    def merge_tree(self,tree,overwrite:bool=False,copy:bool=False):

        if copy == True:
            tree = tree.copy_tree()

        if overwrite == True:
            for node in tree.all_nodes():
                node.parent = self
                self.node_storage.update({node.key:node})

            for att_key in tree.all_attribute_keys():
                self.attribute_storage.update({att_key:tree.ga(att_key)})

        elif overwrite == False:
            for node in tree.all_nodes():
                if node.key not in self.node_storage:
                    node.parent = self
                    self.node_storage.update({node.key:node})

            for att_key in tree.all_attribute_keys():
                if att_key not in self.attribute_storage:
                    self.attribute_storage.update({att_key:tree.ga(att_key)})

        transaction.commit()

    def delete_tree(self,only_node:bool=False):
        self.parent.delete_node(self.key,only_node)
    
    def delete_node(self,node_key:str,only_node:bool=False):

        if str(node_key) in self.node_storage:
            
            if only_node == True:

                for node in self.gn(node_key).all_nodes():
                    self.add_tree(node)

            del self.node_storage[node_key]
        
        transaction.commit()

    def delete_attribute(self,attribute_key):
        if attribute_key in self.attribute_storage:
            del self.attribute_storage[attribute_key]

        transaction.commit()

    def all_attribute_keys(self):
        return list(self.attribute_storage.keys())

    def all_attributes(self):
        return list(self.attribute_storage.values())

    def all_node_keys(self):
        return list(self.node_storage.keys())

    def all_nodes(self):
        return list(self.node_storage.values())

    def copy_tree(self):
        copy_tree = StorageTree(key=self.key)

        for attribute_key in self.all_attribute_keys():
            copy_tree.ga(attribute_key,self.ga(attribute_key))

        if nodes:=self.all_nodes():
            for node in nodes:
                copy_tree.add_tree(node.copy_tree())

        transaction.commit()
        return copy_tree

    def iterate_tree_crone(self):

        if self.all_nodes():
            for node in self.all_nodes():
                yield from node.iterate_tree_crone()
        
        elif not self.all_nodes():
            yield self

    def iterate_tree_level(self,level='deepest'):

        level = self.get_max_depth() if level == 'deepest' else level

        if level > 1:
            for node in self.all_nodes():
                yield from node.iterate_tree_level(level-1)
        
        elif level == 1:
            for node in self.all_nodes():
                yield node

    def get_main_tree(self):
        
        if not isinstance(self.parent,type(None)):
            main_tree = self.parent.get_main_tree()
        else:
            return self

        return main_tree

    def get_max_depth(self,counter=0):
        
        if not self.all_nodes():
            return counter

        max_depth = 0

        for node in self.all_nodes():

            cur_max_depth = node.get_max_depth(counter+1)

            max_depth = cur_max_depth if max_depth < cur_max_depth else max_depth

        return max_depth

    def display(self,counter = 1):
        nodes = self.all_nodes()
        if nodes:
            print('n-'*counter,self.key)
            if self.parent != None:
                print('p-'*counter,self.parent.key)

            for node in nodes:
                print('~n'*counter,counter,node.key)
                print('~a'*counter,node.all_attributes())
                node.display(counter+1)


class PathFileObj(persistent.Persistent):

    def __init__(self,root:str='', file:str='', filepath:str='') -> None:
        
        if filepath != '':
            self.filepath = filepath
        else:
            self.root = root
            self.file = file

    @property
    def root(self) -> str:
        self.check_for_platform()
        return os.sep.join(self.root_path_list)

    @root.setter
    def root(self,root:str) -> None:

        if isinstance(root,str):

            if os.sep in root:
                self.root_path_list = persistent.list.PersistentList(root.split(os.sep))
                    
            else:
                self.root_path_list = persistent.list.PersistentList([root])

        elif isinstance(root,persistent.list.PersistentList):
            self.root_path_list = root

        elif isinstance(root,list):
            self.root_path_list = persistent.list.PersistentList(root)

        transaction.commit()

    @property
    def file(self) -> str:
        self.check_for_platform()
        return self._file

    @file.setter
    def file(self,file) -> None:
        self._file = file
        
        transaction.commit()

    @property
    def filepath(self) -> str:
        self.check_for_platform()
        filepath = self.root_path_list.copy()
        filepath.append(self.file)
        return os.sep.join(filepath)

    @filepath.setter
    def filepath(self,filepath:str) -> None:
        filepath_split = filepath.split(os.sep)
        self.root = filepath_split[:-1]
        self.file = filepath_split[-1]

        transaction.commit()

    def check_for_platform(self):

        cur_platform_details = platform.uname()

        if cur_platform_details.system == 'Windows':
            if '' == self.root_path_list[0]:
                del self.root_path_list[0]

                if 'mnt' == self.root_path_list[0]:
                    del self.root_path_list[0]

                if ':' not in self.root_path_list[0]:
                    self.root_path_list[0] = self.root_path_list[0].upper() + ':'

        elif (cur_platform_details.system == 'Linux') and ('Microsoft' in cur_platform_details.release): 
            self.root_path_list[0] = self.root_path_list[0].lower()

            if ':' in self.root_path_list[0]:
                self.root_path_list[0] = self.root_path_list[0][:-1]
                self.root_path_list = persistent.list.PersistentList(['','mnt']) + self.root_path_list

            if 'mnt' in self.root_path_list[1]:
                self.root_path_list[2] = self.root_path_list[2].lower()

        elif cur_platform_details.system == 'Linux':

            self.root_path_list[0] = self.root_path_list[0].lower()

            if ':' in self.root_path_list[0]:
                self.root_path_list[0] = self.root_path_list[0][:-1]
                self.root_path_list = persistent.list.PersistentList(['']) + self.root_path_list

def filter_node_list(filter_str:str,node_list:list):

    filtered_node_list = []

    for node in node_list:
        if filter_str in node.key:
            filtered_node_list.append(node)

    return filtered_node_list


def add_file(node,filepath_attribute:str='_pfo_audio_wav',root:str='',file:str='',filepath:str=''):

    pfo = PathFileObj(root,file,filepath)
    node.ga(filepath_attribute,pfo)

def read_audio(node,audio_path_attribute='_pfo_audio_wav'):
    from scipy.io.wavfile import read
    from numpy import reshape
    
    filepath = node.ga(audio_path_attribute).filepath

    samplerate, audio  = read(filepath)

    if len(audio.shape) == 1:
        audio = reshape(audio,(audio.shape[0],-1))

    return audio,samplerate


def write_audio(root,node,audio,samplerate,filename=None,pre_parents=list(),audio_path_attribute='_pfo_audio_wav'):
    from scipy.io.wavfile import write

    node_parents = pre_parents.copy()
    node_parents.extend(node.gps())

    filename = '_~_'.join(node_parents) + '.wav' if isinstance(filename,type(None)) else filename

    filepath = os.path.join(root,filename)

    write(filepath,samplerate,audio)

    add_file(node,filepath_attribute=audio_path_attribute,filepath=filepath)

    return 1

def write_raw_audio(root,node,audio,samplerate,filename=None,pre_parents=list(),audio_path_attribute='_pfo_audio_raw'):

    node_parents = pre_parents.copy()
    node_parents.extend(node.gps())

    filename = '_~_'.join(node_parents) + '.raw' if isinstance(filename,type(None)) else filename

    filepath = os.path.join(root,filename)

    raw_audio = audio.tobytes()

    with open(filepath,mode='wb') as file:
        file.write(raw_audio)

    add_file(node,filepath_attribute=audio_path_attribute,filepath=filepath)
    node.ga('_audio_raw_dtype',audio.dtype)
    node.ga('_audio_raw_samplerate',samplerate)

    return 1


def add_g711_a_law_codec(node,path=None,parameter_seperator='_~_'):

    import sox

    if path == None:
        path = node.ga('_pfo_audio_wav').root

    filename =f"{node.ga('_pfo_audio_wav').file[:-4]}{parameter_seperator}g711_a_law.wav"

    codec_node = node.gn('g711_a_law')

    add_file(codec_node,root=path,file=filename)

    audio, samplerate = read_audio(node)

    audio = redatatype_to(audio,float32)

    audio, samplerate = resample_to(audio,samplerate,8000)

    audio_filt = butter_bandpass_filter(audio,samplerate,300,3400,order=5)

    audio_filt = scale_to_other_audio(audio_filt,audio)

    tfm = sox.Transformer()

    tfm.set_output_format(file_type = 'wav',
                          rate = 8000,
                          encoding = 'a-law')

    tfm.build_file(input_array = audio_filt,
                   sample_rate_in = samplerate,
                   output_filepath = codec_node.ga('_pfo_audio_wav').filepath)

    return codec_node

def re_g711_a_law_codec(node,path=None,parameter_seperator='_~_'):

    import sox

    if path == None:
        path = node.ga('_pfo_audio_wav').root

    filename = f"{node.ga('_pfo_audio_wav').file[:-4]}{parameter_seperator}re_g711_a_law.wav"

    re_codec_node = node.gn('re_g711_a_law')
    add_file(re_codec_node,root=path,file=filename)

    tfm = sox.Transformer()

    tfm.set_output_format(encoding = 'signed-integer',
                          file_type = 'wav')

    tfm.build_file(input_filepath = node.ga('_pfo_audio_wav').filepath,
                   output_filepath = re_codec_node.ga('_pfo_audio_wav').filepath)

    return re_codec_node

def add_g711_u_law_codec(node,path=None,parameter_seperator='_~_'):

    import sox

    if path == None:
        path = node.ga('_pfo_audio_wav').root

    filename =f"{node.ga('_pfo_audio_wav').file[:-4]}{parameter_seperator}g711_u_law.wav"

    codec_node = node.gn('g711_u_law')

    add_file(codec_node,root=path,file=filename)

    audio, samplerate = read_audio(node)
    
    audio = redatatype_to(audio,float32)

    audio, samplerate = resample_to(audio,samplerate,8000)

    audio_filt = butter_bandpass_filter(audio,samplerate,300,3400,order=5)

    audio_filt = scale_to_other_audio(audio_filt,audio)

    tfm = sox.Transformer()

    tfm.set_output_format(file_type = 'wav',
                          rate = 8000,
                          encoding = 'u-law')

    tfm.build_file(input_array = audio_filt,
                   sample_rate_in = samplerate,
                   output_filepath = codec_node.ga('_pfo_audio_wav').filepath)

    return codec_node

def re_g711_u_law_codec(node,path=None,parameter_seperator='_~_'):

    import sox

    if path == None:
        path = node.ga('_pfo_audio_wav').root

    filename = f"{node.ga('_pfo_audio_wav').file[:-4]}{parameter_seperator}re_g711_u_law.wav"

    re_codec_node = node.gn('re_g711_u_law')
    add_file(re_codec_node,root=path,file=filename)

    tfm = sox.Transformer()

    tfm.set_output_format(encoding = 'signed-integer',
                          file_type = 'wav')

    tfm.build_file(input_filepath = node.ga('_pfo_audio_wav').filepath,
                   output_filepath = re_codec_node.ga('_pfo_audio_wav').filepath)

    return re_codec_node


def add_gsm_codec(node,path=None,parameter_seperator='_~_'):

    import sox

    if path == None:
        path = node.ga('_pfo_audio_wav').root

    filename =f"{node.ga('_pfo_audio_wav').file[:-4]}{parameter_seperator}gsm.wav"

    codec_node = node.gn('gsm')
    add_file(codec_node,root=path,file=filename)

    audio, samplerate = read_audio(node)

    audio = redatatype_to(audio,float32)

    audio, samplerate = resample_to(audio,samplerate,8000)

    audio_filt = butter_bandpass_filter(audio,samplerate,200,3400,order=5)

    audio_filt = scale_to_other_audio(audio_filt,audio)

    tfm = sox.Transformer()

    tfm.set_output_format(file_type = 'wav',
                          rate = 8000,
                          encoding = 'gsm-full-rate')

    tfm.build_file(input_array = audio_filt,
                   sample_rate_in = samplerate,
                   output_filepath = codec_node.ga('_pfo_audio_wav').filepath)

    return codec_node


def re_gsm_codec(node,path=None,parameter_seperator='_~_'):

    import sox

    if path == None:
        path = node.ga('_pfo_audio_wav').root

    filename = f"{node.ga('_pfo_audio_wav').file[:-4]}{parameter_seperator}re_gsm.wav"

    re_codec_node = node.gn('re_gsm')

    add_file(re_codec_node,root=path,file=filename)

    tfm = sox.Transformer()

    tfm.set_output_format(encoding = 'signed-integer',
                          file_type = 'wav')

    tfm.build_file(input_filepath= node.ga('_pfo_audio_wav').filepath,
                   output_filepath = re_codec_node.ga('_pfo_audio_wav').filepath)


    return re_codec_node


def add_g722_codec(node,path=None,parameter_seperator='_~_'):
    import subprocess

    if path == None:
        path = node.ga('_pfo_audio_wav').root

    filename =f"{node.ga('_pfo_audio_wav').file[:-4]}{parameter_seperator}g722.wav"

    codec_node = node.gn('g722')

    add_file(codec_node,root=path,file=filename)

    audio, samplerate = read_audio(node)

    audio, samplerate = resample_to(audio,samplerate,16000)

    filt_audio = butter_bandpass_filter(audio,samplerate,50,7000,order=5)

    filt_audio = scale_to_other_audio(filt_audio,audio)

    rel_path = PathFileObj(filepath=os.path.realpath(__file__)).root

    cur_node = node.copy_tree()

    returned = write_audio(rel_path,cur_node,filt_audio,samplerate)

    if returned:

        cur_file_path = cur_node.ga('_pfo_audio_wav').filepath

        command = ['ffmpeg', '-y', '-i', cur_file_path, '-acodec', 'g722', codec_node.ga("_pfo_audio_wav").filepath]

        returned = subprocess.check_call(command)
        
        os.remove(cur_file_path)

    return codec_node

def re_g722_codec(node,path=None,parameter_seperator='_~_'):
    import subprocess

    if path == None:
        path = node.ga('_pfo_audio_wav').root

    filename =f"{node.ga('_pfo_audio_wav').file[:-4]}{parameter_seperator}re_g722.wav"

    re_codec_node = node.gn('re_g722')

    add_file(re_codec_node,root=path,file=filename)

    command = ['ffmpeg', '-y', '-acodec', 'g722',  '-i', node.ga('_pfo_audio_wav').filepath, '-acodec', 'pcm_s16le', re_codec_node.ga('_pfo_audio_wav').filepath]

    returned = subprocess.check_call(command)

    return re_codec_node


def save_to_file(oject,path=None):
    import pickle

    if path is None: path = 'saved_obj'
    
    with open(f'{path}.pyobj',mode='wb') as file:

        pickle.dump(oject,file)


def load_from_file(path):
    import pickle

    with open(path,mode='rb') as file:

        oject = pickle.load(file)

    return oject


def load_files_directory(audio_tree:StorageTree,dir_path:str,file_type_pfo_name:str='_pfo_audio_wav',file_extension:str='wav'):

    root_folder = dir_path.split(os.sep)[-1]

    for root,_,files in os.walk(dir_path):
        if files:
            root_split = root.split(os.path.sep)
            root_idx = root_split.index(root_folder)
            last_node = audio_tree.gns(root_split[root_idx:])

            for file in files:

                if file.endswith(f'.{file_extension}'):

                    cur_node = last_node.gn(file)

                    cur_node.ga(file_type_pfo_name,PathFileObj(root,file))


def load_files_folder(audio_tree:str,folder_path:str,parameter_seperator:str='_~_',filepath_attribute:str='_pfo_audio_wav',file_extension:str='wav'):

    folder_file_list = os.listdir(folder_path)

    for file in folder_file_list:

        if file.endswith(f'.{file_extension}'):

            cur_file = file[0:-4]

            file_node_names = cur_file.split(parameter_seperator)

            cur_node = audio_tree.gns(file_node_names)

            add_file(cur_node,filepath_attribute=filepath_attribute,root=folder_path,file=file)
    

def node_process_cruncher(function,node_list:list[StorageTree],other_args_dic:dict={},processes=os.cpu_count()-1):
    from multiprocessing import Pool
    from itertools import repeat

    main_tree = node_list[0].get_main_tree()

    copy_node_list = []

    for node in node_list:
        copy_node = node.copy_tree()
        copy_node.ga('parents',node.gps())
        copy_node.parent = None
        copy_node_list.append(copy_node)

    kwargs_iter = repeat(other_args_dic)

    #maxtasksperchild=1000

    with Pool(processes=processes) as pool:
        results = starmap_with_kwargs(pool,function,copy_node_list,kwargs_iter)

    for result_node in results:
        main_tree.gns(result_node.ga('parents')).merge_tree(result_node,overwrite=True)

        main_tree.gns(result_node.ga('parents')).delete_attribute('parents')


def starmap_with_kwargs(pool, fn, node_list, kwargs_iter):
    from itertools import repeat
    args_for_starmap = zip(repeat(fn), node_list, kwargs_iter)

    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, node, kwargs):
    return fn(node,**kwargs)

def tree_to_dataframe(tree:StorageTree,attribute_list):

    import pandas as pd 

    dataframe = pd.DataFrame()

    last_pidx = 0

    for pidx in range(tree.get_max_depth()):
        parent_name = f'TreeLevel_{pidx+1}'
        dataframe.insert(pidx,column=parent_name,value=[])
        last_pidx = pidx

    for node in tree.iterate_tree_crone():
        tree_key = tree.key
        node_parents = node.gps()
        cur_idx = node_parents.index(tree_key) + 1 if tree_key in node_parents else 0
        node_parents = node_parents[cur_idx:]
        dataframe = dataframe.append(pd.DataFrame([node_parents], columns=dataframe.columns[:len(node_parents)]), ignore_index=True)


    last_pidx += 1
    for att in attribute_list:
        att_list = []
        for node in tree.iterate_tree_crone():

            cur_att = node.ga(att)
            if isinstance(cur_att,ndarray):
                cur_att = cur_att.tolist()

            att_list.append(cur_att)

        dataframe.insert(last_pidx,column=att,value=att_list)
        last_pidx += 1

    return dataframe


def test_counter(node,offset):

    node.ga('c',node.ga('a')+node.ga('b')+offset)

    return node

if __name__ == '__main__':

    tree = open_StorageTree_DB('test','test')

    nodes_till = ['hi','mein','name','till']
    nodes_frieda = ['hi','mein','name','frieda']

    till_node = tree.gns(nodes_till)
    frieda_node = tree.gns(nodes_frieda)

    till_node.ga('a',1)
    till_node.ga('b',2)

    frieda_node.ga('a',1)
    frieda_node.ga('b',2)

    df = tree_to_dataframe(tree.gn('hi'),['a','b'])
    print('df: ', df)





    


