import os
import platform
from numpy import *
#from utilFunctions import *

import ZODB
import ZODB.FileStorage

import persistent
import persistent.list

import transaction
from BTrees._OOBTree import OOBTree

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

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self,parent):

        self._parent = parent


    def ga(self,attribute_key:str,attribute:str=None):

        if str(attribute_key) not in self.attribute_storage:

            self.attribute_storage.update({str(attribute_key):attribute})

        elif not isinstance(attribute,type(None)):

            self.attribute_storage.update({str(attribute_key):attribute})

        return self.attribute_storage[str(attribute_key)]

    def gna(self,node_keys,attribute_key:str,attribute:str=None):

        cur_node = self.gns(node_keys)

        return cur_node.ga(attribute_key,attribute)

    def gn(self,node_key:str):

        if str(node_key) not in self.node_storage:

            self.node_storage.update({str(node_key):StorageTree(key=node_key,parent=self)})

        return self.node_storage[str(node_key)]

    def gns(self,node_keys):

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

    def get_nodes(self,node_keys,including:bool=True):

        if including == True:
            return [self.gn(node_key) for node_key in node_keys]
        
        elif including == False:
            return [self.gn(node_key) for node_key in list(self.all_node_keys()) 
                    if node_key not in node_keys]
    
    def add_tree(self,tree,copy:bool=False,unique_key:bool=False):

        if unique_key == True:
            new_key_list = tree.gps()
            tree_key = '_~_'.join(new_key_list)
        else:
            tree_key = tree.key

        if copy == True:
            c_tree = tree.copy_tree()
            c_tree.key = tree_key
            tree = c_tree

        self.node_storage.update({c_tree.key:tree})
        tree.parent = self

    def add_trees(self,tree_list,copy:bool=False,unique_key:bool=False):

        for tree in tree_list:

            self.add_tree(tree,copy,unique_key)

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

    def delete_tree(self,only_node:bool=False):
        self.parent.delete_node(self.key,only_node)
    
    def delete_node(self,node_key:str,only_node:bool=False):

        if str(node_key) in self.node_storage:
            
            if only_node == True:

                for node in self.gn(node_key).all_nodes():
                    self.add_tree(node)

            del self.node_storage[node_key]

    def delete_attribute(self,attribute_key):
        if attribute_key in self.attribute_storage:
            del self.attribute_storage[attribute_key]

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
                copy_tree.add_tree(node,copy=True)

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

    def _buffered_iteration(self,iter_obj,buffer_size):

        run = True

        while run:
            node_list = []

            run_2 = True

            while run_2:
                
                try:
                    next_node = next(iter_obj)

                except StopIteration:
                    run = False
                    break
            
                node_list.append(next_node)

                if len(node_list) == buffer_size:
                    run_2 = False
                    yield node_list
                    
            if not run:
                if node_list:
                    yield node_list

    def iterate_tree_level_buffered(self,buffer_size:int=100_000,level='deepest'):

        iter_obj = self.iterate_tree_level(level=level)

        return self._buffered_iteration(iter_obj=iter_obj,buffer_size=buffer_size)

    def iterate_tree_crone_buffered(self,buffer_size:int=100_000):

        iter_obj = self.iterate_tree_crone()

        return self._buffered_iteration(iter_obj=iter_obj,buffer_size=buffer_size)
                

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

    def unconnect(self):

        return StorageTreeUnconnected(self)

class StorageTreeUnconnected():

    def __init__(self,tree:StorageTree) -> None:
        
        self.parents = tree.gps()

class StorageTreeDatabase():

    def __init__(self,path:str='',
                 name:str='StorageTreeDB',
                 new:bool=False,
                 read_only:bool=False) -> None:

        self.path = path
        self.name = name
        self.db = self.setup_StorageTree_DB(new=new,read_only=read_only)
        self.conn_dict = {}

    def setup_StorageTree_DB(self,new:bool=False,read_only=False):

        os.makedirs(self.path,exist_ok=True)

        pfo = PathFileObj(root=self.path,file=self.name)

        storage = ZODB.FileStorage.FileStorage(pfo.filepath,
                                               create=new,
                                               read_only=read_only)
        db = ZODB.DB(storage)

        if new:
            with db.transaction() as conn:
                conn.root.stt = StorageTree(key=self.name)

        return db
    
    def open_connection(self,
                        unconnected_tree:StorageTreeUnconnected=None,
                        connection_id:str = 'standard'):

        transaction_manager = transaction.TransactionManager()

        conn = self.db.open(transaction_manager=transaction_manager)

        self.conn_dict.update({connection_id:[conn, transaction_manager]})

        stt = conn.root.stt

        if not isinstance(unconnected_tree,type(None)):
            stt = stt.gns(unconnected_tree.parents)

        return stt
    
    def close_connection(self,connection_id:str = 'standard'):
        if self.conn_dict:
            conn,_ = self.conn_dict[connection_id]
            conn.close()
            del self.conn_dict[connection_id]
    
    def commit(self,connection_id:str = 'standard'):
        if self.conn_dict:
            _,transaction_manager = self.conn_dict[connection_id]
            transaction_manager.commit()

    def close(self):
        if self.conn_dict:
            key_list = list(self.conn_dict.keys())
            for key in key_list:
                self.close_connection(connection_id=key)

        self.db.close()
    
    def load_files_directory(self,
                             dir_path:str,
                             parameter_seperator:str=None,
                             filepath_attribute:str='_pfo_audio_wav',
                             file_extension:str='wav'):
        
        if 'standard' in self.conn_dict:
            tree = self.conn_dict['standard'][0].root.stt
        else:
            tree = self.open_connection()

        root_folder = dir_path.split(os.sep)[-1]

        for root,_,files in os.walk(dir_path):
            if files:
                root_split = root.split(os.path.sep)
                root_idx = root_split.index(root_folder)
                last_node = tree.gns(root_split[root_idx:])

                for file in files:

                    if file.endswith(f'.{file_extension}'):

                        if parameter_seperator is None:
                            cur_node = last_node.gn(file[0:-4])
                        else:
                            cur_node = last_node.gns(file[0:-4].split(sep=parameter_seperator))

                        add_file(cur_node,filepath_attribute=filepath_attribute,root=root,file=file)

                        self.commit()


        self.close_connection()

    def load_files_folder(self,
                          folder_path:str,
                          parameter_seperator:str='_~_',
                          filepath_attribute:str='_pfo_audio_wav',
                          file_extension:str='wav'):
        
        if 'standard' in self.conn_dict:
            tree = self.conn_dict['standard'][0].root.stt
        else:
            tree = self.open_connection()

        folder_file_list = os.listdir(folder_path)

        for file in folder_file_list:

            if file.endswith(f'.{file_extension}'):

                cur_file = file[0:-4]

                file_node_names = cur_file.split(parameter_seperator)

                cur_node = tree.gns(file_node_names)

                add_file(cur_node,filepath_attribute=filepath_attribute,root=folder_path,file=file)

                self.commit()

        self.close_connection()

def load_small_files_directory(tree:StorageTree,
                               dir_path:str,
                               parameter_seperator:str=None,
                               filepath_attribute:str='_pfo_audio_wav',
                               file_extension:str='wav'):

    root_folder = dir_path.split(os.sep)[-1]

    for root,_,files in os.walk(dir_path):
        if files:
            root_split = root.split(os.path.sep)
            root_idx = root_split.index(root_folder)
            last_node = tree.gns(root_split[root_idx:])

            for file in files:

                if file.endswith(f'.{file_extension}'):

                    if parameter_seperator is None:
                        cur_node = last_node.gn(file[0:-4])
                    else:
                        cur_node = last_node.gns(file[0:-4].split(sep=parameter_seperator))

                    add_file(cur_node,filepath_attribute=filepath_attribute,root=root,file=file)

def load_small_files_folder(tree:StorageTree,
                            folder_path:str,
                            parameter_seperator:str='_~_',
                            filepath_attribute:str='_pfo_audio_wav',
                            file_extension:str='wav'):

    folder_file_list = os.listdir(folder_path)

    for file in folder_file_list:

        if file.endswith(f'.{file_extension}'):

            cur_file = file[0:-4]

            file_node_names = cur_file.split(parameter_seperator)

            cur_node = tree.gns(file_node_names)

            add_file(cur_node,filepath_attribute=filepath_attribute,root=folder_path,file=file)
    
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

    @property
    def file(self) -> str:
        self.check_for_platform()
        return self._file

    @file.setter
    def file(self,file) -> None:
        self._file = file

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

def filter_node_list(filter_str:str,node_list:list,invers=False):
    import re

    filtered_node_list = []

    for node in node_list:
        if invers:
            if (re.search(filter_str,node.key) is None):
                filtered_node_list.append(node)    
        else:
            if not (re.search(filter_str,node.key) is None):
                filtered_node_list.append(node)

    return filtered_node_list

def add_file(node,filepath_attribute:str='_pfo_audio_wav',root:str='',file:str='',filepath:str=''):

    pfo = PathFileObj(root,file,filepath)
    node.ga(filepath_attribute,pfo)

def rename_file(node,filepath_attribute:str='_pfo_audio_wav',new_root:str='',new_file:str='',new_filepath:str=''):

    pfo = node.ga(filepath_attribute)

    pfo_filepath = pfo.filepath

    if new_root == '':
        pfo.file = new_file
    elif new_file == '':
        pfo.root = new_root
    else:
        pfo.filepath = new_filepath

    os.replace(pfo_filepath,pfo.filepath)

class FileReader():

    @staticmethod
    def load_audio(filepath):
        from scipy.io import wavfile
        from numpy import reshape

        samplerate, audio  = wavfile.read(filepath)

        if len(audio.shape) == 1:
            audio = reshape(audio,(audio.shape[0],-1))

        return audio,samplerate

    @staticmethod
    def read_audio_wav(node,filepath_attribute='_pfo_audio_wav'):

        return FileReader.load_audio(node.ga(filepath_attribute).filepath)
    
    @staticmethod
    def read_table_txt(node,filepath_attribute='_pfo_table_txt',header='infer'):
        import pandas as pd

        filepath = node.ga(filepath_attribute).filepath

        csv_df = pd.read_table(filepath,header=header)

        return csv_df
    
    @staticmethod
    def read_array_npy(node,filepath_attribute='_pfo_array_npy'):
        from numpy import load

        filepath = node.ga(filepath_attribute).filepath

        if not '.npy' in filepath:
            filepath = filepath + '.npy'

        arr = load(filepath)

        return arr
    
    def read_array_mat(node,filepath_attribute='_pfo_array_mat'):
        from scipy.io import loadmat

        filepath = node.ga(filepath_attribute).filepath

        arr = loadmat(filepath)

        return arr


class FileWriter():

    @staticmethod
    def setup_tree_directories(root:str,
                               node:str,
                               pre_parents:list):
        
        node_parents = pre_parents.copy()
        node_parents.extend(node.gps())

        new_root = root

        for parent in node_parents:

            path = os.path.join(new_root,parent)

            os.makedirs(path,exist_ok=True)
            
            new_root = path

        return new_root

    @staticmethod
    def create_filename(pre_parents:list,
                        node:StorageTree,
                        file_type:str,
                        parameter_seperator:str):

        node_parents = pre_parents.copy()
        node_parents.extend(node.gps())

        filename = parameter_seperator.join(node_parents) + '.' + file_type 

        return filename

    @staticmethod
    def write_audio_wav_into_tree_directories(root:str,
                                              node:StorageTree,
                                              audio:ndarray,
                                              samplerate:int,
                                              filename=None,
                                              pre_parents:list=list(),
                                              filepath_attribute:str='_pfo_audio_wav'):
        
        root = FileWriter.setup_tree_directories(root=root,node=node,pre_parents=pre_parents)

        FileWriter.write_audio_wav(root=root,
                             node=node,
                             audio=audio,
                             samplerate=samplerate,
                             filename=filename,
                             pre_parents=pre_parents,
                             filepath_attribute=filepath_attribute)
    
    @staticmethod
    def write_audio_raw_into_tree_directories(root:str,
                                              node:StorageTree,
                                              audio:ndarray,
                                              samplerate:int,
                                              filename=None,
                                              pre_parents:list=list(),
                                              filepath_attribute:str='_pfo_audio_raw',
                                              parameter_seperator:str='_~_'):
        
        root = FileWriter.setup_tree_directories(root=root,node=node,pre_parents=pre_parents)

        FileWriter.write_audio_raw(root=root,
                             node=node,
                             audio=audio,
                             samplerate=samplerate,
                             filename=filename,
                             pre_parents=pre_parents,
                             filepath_attribute=filepath_attribute,
                             parameter_seperator=parameter_seperator)
    
    @staticmethod
    def write_array_numpy_into_tree_directories(root:str,
                                                node:StorageTree,
                                                array:ndarray,
                                                filename=None,
                                                pre_parents:list=list(),
                                                filepath_attribute:str='_pfo_array_npy',
                                                parameter_seperator:str='_~_'):
        
        root = FileWriter.setup_tree_directories(root=root,node=node,pre_parents=pre_parents)

        FileWriter.write_array_npy(root=root,
                             node=node,
                             array=array,
                             filename=filename,
                             pre_parents=pre_parents,
                             filepath_attribute=filepath_attribute,
                             parameter_seperator=parameter_seperator)

    @staticmethod
    def write_audio_wav(root:str,
                        node:StorageTree,
                        audio:ndarray,
                        samplerate:int,
                        filename:str=None,
                        pre_parents:list=list(),
                        filepath_attribute:str='_pfo_audio_wav',
                        parameter_seperator:str='_~_'):
        
        from scipy.io.wavfile import write

        if isinstance(filename,type(None)):
            filename = FileWriter.create_filename(pre_parents=pre_parents,
                                                  node=node,
                                                  file_type='wav',
                                                  parameter_seperator=parameter_seperator)

        filepath = os.path.join(root,filename)

        write(filepath,samplerate,audio)

        add_file(node,filepath_attribute=filepath_attribute,filepath=filepath)

        return 1

    @staticmethod
    def write_audio_raw(root:str,
                        node:StorageTree,
                        audio:ndarray,
                        samplerate:int,
                        filename:str=None,
                        pre_parents=list(),
                        filepath_attribute:str='_pfo_audio_raw',
                        parameter_seperator:str='_~_'):

        if isinstance(filename,type(None)):
            filename = FileWriter.create_filename(pre_parents=pre_parents,
                                                  node=node,
                                                  file_type='raw',
                                                  parameter_seperator=parameter_seperator)

        filepath = os.path.join(root,filename)

        raw_audio = audio.tobytes()

        with open(filepath,mode='wb') as file:
            file.write(raw_audio)

        add_file(node,filepath_attribute=filepath_attribute,filepath=filepath)
        node.ga('_audio_raw_dtype',audio.dtype)
        node.ga('_audio_raw_samplerate',samplerate)

        return 1

    @staticmethod
    def write_array_npy(root,
                        node,
                        array,
                        filename=None,
                        pre_parents=list(),
                        filepath_attribute:str='_pfo_array_npy',
                        parameter_seperator:str='_~_'):
        
        from numpy import save

        if isinstance(filename,type(None)):
            filename = FileWriter.create_filename(pre_parents=pre_parents,
                                                  node=node,
                                                  file_type='npy',
                                                  parameter_seperator=parameter_seperator)

        filepath = os.path.join(root,filename)

        save(filepath,array)

        add_file(node,filepath_attribute=filepath_attribute,filepath=filepath)

        return 1

    @staticmethod
    def write_table_txt(root:str,
                        node:StorageTree,
                        dataframe,
                        header=True,
                        index:bool=False,
                        filename:str=None,
                        pre_parents=list(),
                        filepath_attribute:str='_pfo_table_txt',
                        parameter_seperator:str='_~_'):

        if isinstance(filename,type(None)):
            filename = FileWriter.create_filename(pre_parents=pre_parents,
                                                  node=node,
                                                  file_type='txt',
                                                  parameter_seperator=parameter_seperator)

        filepath = os.path.join(root,filename)

        dataframe.to_csv(filepath,header=header,index=index)

        add_file(node,filepath_attribute=filepath_attribute,filepath=filepath)

        return 1


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
    

def node_process_cruncher(function,node_list,other_args_dic:dict={},processes=os.cpu_count()-1):
    from multiprocessing import Pool,get_context
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

    with get_context('spawn').Pool(processes=processes) as pool:
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
        dataframe = dataframe._append(pd.DataFrame([node_parents], columns=dataframe.columns[:len(node_parents)]), ignore_index=True)

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

def node_list_to_dataframe(node_list:list,attribute_list):

    import pandas as pd 

    dataframe = pd.DataFrame()

    last_pidx = 0

    for pidx in range(len(node_list[0].gps())):
        parent_name = f'TreeLevel_{pidx+1}'
        dataframe.insert(pidx,column=parent_name,value=[])
        last_pidx = pidx

    for node in node_list:
        node_parents = node.gps()
        dataframe = dataframe._append(pd.DataFrame([node_parents], columns=dataframe.columns[:len(node_parents)]), ignore_index=True)

    last_pidx += 1
    for att in attribute_list:
        att_list = []
        for node in node_list:

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

    tree_db = StorageTreeDatabase(path='test',new=True)

    tree = tree_db.open_connection()

    node = tree.gns(['hi','my','name','is','Till'])
    node.ga('hi','Peter')
    node = tree.gns(['hi','my','name','is','Frieda'])
    node.ga('hi','peeter')
    node = tree.gns(['hi','my','name','is','Till1'])
    node.ga('hi','Peter')
    node = tree.gns(['hi','my','name','is','Frieda1'])
    node.ga('hi','peeter')
    node = tree.gns(['hi','my','name','is','Till2'])
    node.ga('hi','Peter')
    node = tree.gns(['hi','my','name','is','Till3'])
    node.ga('hi','Peter')
    
    tree_db.commit()

    for node_list in tree.iterate_tree_level_buffered(buffer_size=3,level=5):
        print(node_list)
        for node in node_list:
            print(node.gps())





    

