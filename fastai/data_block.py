from .torch_core import *
from .basic_data import *
from .layers import *
from numbers import Integral

__all__ = ['ItemList', 'CategoryList', 'MultiCategoryList', 'MultiCategoryProcessor', 'LabelList', 'ItemLists', 'get_files',
           'PreProcessor', 'LabelLists', 'FloatList', 'CategoryProcessor', 'EmptyLabelList', 'MixedItem', 'MixedProcessor',
           'MixedItemList']

def _decode(df):
    return np.array([[df.columns[i] for i,t in enumerate(x) if t==1] for x in df.values], dtype=np.object)

def _maybe_squeeze(arr): return (arr if is1d(arr) else np.squeeze(arr))

def _path_to_same_str(p_fn):
    "path -> str, but same on nt+posix, for alpha-sort only"
    s_fn = str(p_fn)
    s_fn = s_fn.replace('\\','.')
    s_fn = s_fn.replace('/','.')
    return s_fn

def _get_files(parent, p, f, extensions):
    """
    `_get_files`: 
        1. extract a list of files from `f` the list
        2. files must have suffix in the extensions.
        3. return a list of Path objects

    ----inputs 
    `parent`: path_data
    `p`: subfolder path
    `f`: a list of files from `p`
    `extensions`: a list of suffix

    ----cases
    1. hidden files are ignored
    2. all suffix can be lower or upper cases

    """
    # create a Path object for p 
    p = Path(p)#.relative_to(parent)
    # if only one suffix, then put it into a list
    if isinstance(extensions,str): extensions = [extensions]
    # make sure all suffix are in lower case
    low_extensions = [e.lower() for e in extensions] if extensions is not None else None
    # make sure the files are not hidden and suffix are in extensions
    res = [p/o for o in f if not o.startswith('.')
           and (extensions is None or f'.{o.split(".")[-1].lower()}' in low_extensions)]
    # res is a list of Path objects
    return res

def get_files(path:PathOrStr, extensions:Collection[str]=None, recurse:bool=False,
              include:Optional[Collection[str]]=None, presort:bool=False)->FilePathList:
    """
    Return list of files in `path` that have a suffix in `extensions`; optionally `recurse`.
    ----why
    `get_files`: 
        1. return a list of files extracted from a folder path
        2. usually used by `from_folder` etc, not directly used.

    ----inputs
    `cls`: @classmethod uses cls instead of self
    `path`: folder path
    `extensions`: a list of file suffix
    `recurse`: whether to extract subfolders
    `include`: a list of folders to extract files from
    `processor`: a list of processors 
    `presort`: whether to pre-sort files??
    `**kwargs`: a dict of additional args with values,
                passing onto next level functions 

    ----internals
    `_get_files`: 
        1. extract files from the list `f` from the path `path/p`
        2. all the files are with suffix in `extensions`

    ----cases
    1. choose to extract from subfolders = `recursive=True`
    2. choose to only use a group of subfolders = `include=[some subfolders names]`
    3. hidden folders and files are not extracted
    """
    # if recurse == True
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path)):
            # skip hidden dirs
            # if `include` has a list of folders, d will only contain `include`
            if include is not None and i==0:  d[:] = [o for o in d if o in include]
            # otherwise, d will just exclude hidden folders and files
            else:                             d[:] = [o for o in d if not o.startswith('.')]
            # get a list of files with suffix out of `extensions`, 
            # and these files are from `f` (the list of files)
            # add such a list into `res`
            res += _get_files(path, p, f, extensions)

        # if `presort==True`, then sort files from image number from small to large. 
        if presort: res = sorted(res, key=lambda p: _path_to_same_str(p), reverse=False)
        return res
    # otherwise => recursive==False, only extract files in the immediate folder
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, path, f, extensions)
        if presort: res = sorted(res, key=lambda p: _path_to_same_str(p), reverse=False)
        return res

class PreProcessor():
    "Basic class for a processor that will be applied to items at the end of the data block API."
    def __init__(self, ds:Collection=None):  self.ref_ds = ds
    def process_one(self, item:Any):    return item    

    def process(self, ds:Collection):       
        """
        ds: an object of ItemList
        using `process_one` to process each item of `ds.items` and put them into an array,
        and then assign back to `ds.items`.
        """
        ds.items = array([self.process_one(item) for item in ds.items])

PreProcessors = Union[PreProcessor, Collection[PreProcessor]]
fastai_types[PreProcessors] = 'PreProcessors'

class ItemList():
    "A collection of items with `__len__` and `__getitem__` with `ndarray` indexing semantics."
    _bunch,_processor,_label_cls,_square_show,_square_show_res = DataBunch,None,None,False,False

    def __init__(self, items:Iterator, path:PathOrStr='.', label_cls:Callable=None, inner_df:Any=None,
                 processor:PreProcessors=None, x:'ItemList'=None, ignore_empty:bool=False):
        """
        ----why
        `ItemList.__init__`
            1. prepare the properties for a list of items for calling methods

        ----inputs and internals
            1. `_` and `__` prefix are internals
            2. path properties: `path`, `num_parts`
            3. data prperties: `items`, `x`, `ignore_empty`, `inner_df`
            4. label properties: `label_cls`, `_label_list`
            5. processor properties: `processor`
            6. split properties: `_split`
            7. new-ItemList properites: `copy_new`
            8. calling `__post_init__`
        """
        # add `path` as a property
        self.path = Path(path)
        # add `num_parts` of the path as a property
        self.num_parts = len(self.path.parts)
        # add `items`, `x`, `ignore_empty` as properties to handle data
        self.items,self.x,self.ignore_empty = items,x,ignore_empty
        # make sure `items` to be an array
        if not isinstance(self.items,np.ndarray): self.items = array(self.items, dtype=object)
        # add `inner_df`, `processors` and `label_cls` as properties
        self.label_cls,self.inner_df,self.processor = ifnone(label_cls,self._label_cls),inner_df,processor
        # add `LabelList`, `ItemLists` to  internal properties
        # `_label_list`, `_split` respectively
        self._label_list,self._split = LabelList,ItemLists
        # set property `copy_new` for creating new `Itemlist`
        self.copy_new = ['x', 'label_cls', 'path']
        # calling internal function `__post__init()`
        self.__post_init__()

    def __post_init__(self): pass
    def __len__(self)->int: 
        """
        `ItemList.__len__`:
            1. equivalent to `len(ItemList)`
            2. return `len(ItemList.items)` or 1
        """
        return len(self.items) or 1

    def get(self, i)->Any:
        """
        "Subclass if you want to customize how to create item `i` from `self.items`."
        ----why
        `ItemList.get`:
            1. to get individual item from `self.items`
            2. different subclasses should/may have to 
               have different version of it
        """
        return self.items[i]
    def __repr__(self)->str:
        """
        `ItemList.__repr__`:
            1. how `ItemList` object should be printed out
            2. print out more than 5 items
            3. each item is accessed by `self[i]` or `self.__getitem__(i)`
            4. also print out info on class, length, and those items above

        ----internals
        `show_some`: print out items 
        """
        items = [self[i] for i in range(min(5,len(self.items)))]
        return f'{self.__class__.__name__} ({len(self.items)} items)\n{show_some(items)}\nPath: {self.path}'

    def process(self, processor:PreProcessors=None):
        "Apply `processor` or `self.processor` to `self`."
        if processor is not None: self.processor = processor
        self.processor = listify(self.processor)
        for p in self.processor: p.process(self)
        return self
        """
        How does `ItemList.process` or `y.process(yp)` work?
        `self` is an object of `ItemList` or its subclasses
        `processor` is one or more `PreProcessors` objects
        Behind the scenes, we put all of `processor` into a list and apply them all to the `self`.
        """

    def process_one(self, item:ItemBase, processor:PreProcessors=None):
        "Apply `processor` or `self.processor` to `item`."
        if processor is not None: self.processor = processor
        self.processor = listify(self.processor)
        for p in self.processor: item = p.process_one(item)
        return item

    def analyze_pred(self, pred:Tensor):
        "Called on `pred` before `reconstruct` for additional preprocessing."
        return pred

    def reconstruct(self, t:Tensor, x:Tensor=None):
        "Reconstruct one of the underlying item for its data `t`."
        return self[0].reconstruct(t,x) if has_arg(self[0].reconstruct, 'x') else self[0].reconstruct(t)

    def new(self, items:Iterator, processor:PreProcessors=None, **kwargs)->'ItemList':
        "Create a new `ItemList` from `items`, keeping the same attributes."
        processor = ifnone(processor, self.processor)
        copy_d = {o:getattr(self,o) for o in self.copy_new}
        kwargs = {**copy_d, **kwargs}
        return self.__class__(items=items, processor=processor, **kwargs)

    def add(self, items:'ItemList'):
        self.items = np.concatenate([self.items, items.items], 0)
        if self.inner_df is not None and items.inner_df is not None:
            self.inner_df = pd.concat([self.inner_df, items.inner_df])
        else: self.inner_df = self.inner_df or items.inner_df
        return self

    def __getitem__(self,idxs:int)->Any:
        """
        `ItemList.__getitem__`:
            1. when `idxs` is a single number, return `self.get(idxs)`
            2. when a range of numbers, return a new `ItemList` of those items
        """
        idxs = try_int(idxs)
        if isinstance(idxs, Integral): return self.get(idxs)
        else: return self.new(self.items[idxs], inner_df=index_row(self.inner_df, idxs))

    @classmethod
    def from_folder(cls, path:PathOrStr, extensions:Collection[str]=None, recurse:bool=True,
                    include:Optional[Collection[str]]=None, processor:PreProcessors=None, presort:Optional[bool]=False, **kwargs)->'ItemList':
        """Create an `ItemList` in `path` from the filenames that have a suffix in `extensions`.
        `recurse` determines if we search subfolders.

        ----why
        `ItemList.from_folder`:
        1. extract a list of files from a folder, and 
        2. pass onto `ItemList.__init__` to instantiate an object
        
        ----inputs
        `cls`: @classmethod uses cls instead of self
        `path`: folder path
        `extensions`: a list of file suffix
        `recurse`: whether to extract subfolders
        `include`: a list of folders to extract files from
        `processor`: a list of processors 
        `presort`: whether to pre-sort files, e.g. MNIST_TINY
        `**kwargs`: a dict of additional args with values,
                    passing onto next level functions  
       
        ----internals
        `get_files`: extract files of folders into a list
        `cls`: `ItemList.__init__`
        """
        path = Path(path)
        return cls(get_files(path, extensions, recurse=recurse, include=include, presort=presort), path=path, processor=processor, **kwargs)

    @classmethod
    def from_df(cls, df:DataFrame, path:PathOrStr='.', cols:IntsOrStrs=0, processor:PreProcessors=None, **kwargs)->'ItemList':
        "Create an `ItemList` in `path` from the inputs in the `cols` of `df`."
        inputs = df.iloc[:,df_names_to_idx(cols, df)]
        assert inputs.isna().sum().sum() == 0, f"You have NaN values in column(s) {cols} of your dataframe, please fix it."
        res = cls(items=_maybe_squeeze(inputs.values), path=path, inner_df=df, processor=processor, **kwargs)
        return res

    @classmethod
    def from_csv(cls, path:PathOrStr, csv_name:str, cols:IntsOrStrs=0, delimiter:str=None, header:str='infer',
                 processor:PreProcessors=None, **kwargs)->'ItemList':
        """Create an `ItemList` in `path` from the inputs in the `cols` of `path/csv_name`"""
        df = pd.read_csv(Path(path)/csv_name, delimiter=delimiter, header=header)
        return cls.from_df(df, path=path, cols=cols, processor=processor, **kwargs)

    def _relative_item_path(self, i): return self.items[i].relative_to(self.path)
    def _relative_item_paths(self):   return [self._relative_item_path(i) for i in range_of(self.items)]

    def use_partial_data(self, sample_pct:float=0.01, seed:int=None)->'ItemList':
        "Use only a sample of `sample_pct`of the full dataset and an optional `seed`."
        if seed is not None: np.random.seed(seed)
        rand_idx = np.random.permutation(range_of(self))
        cut = int(sample_pct * len(self))
        return self[rand_idx[:cut]]

    def to_text(self, fn:str):
        "Save `self.items` to `fn` in `self.path`."
        with open(self.path/fn, 'w') as f: f.writelines([f'{o}\n' for o in self._relative_item_paths()])

    def filter_by_func(self, func:Callable)->'ItemList':
        "Only keep elements for which `func` returns `True`."
        self.items = array([o for o in self.items if func(o)])
        return self

    def filter_by_folder(self, include=None, exclude=None):
        "Only keep filenames in `include` folder or reject the ones in `exclude`."
        include,exclude = listify(include),listify(exclude)
        def _inner(o):
            if isinstance(o, Path): n = o.relative_to(self.path).parts[0]
            else: n = o.split(os.path.sep)[len(str(self.path).split(os.path.sep))]
            if include and not n in include: return False
            if exclude and     n in exclude: return False
            return True
        return self.filter_by_func(_inner)

    def filter_by_rand(self, p:float, seed:int=None):
        "Keep random sample of `items` with probability `p` and an optional `seed`."
        if seed is not None: np.random.seed(seed)
        return self.filter_by_func(lambda o: rand_bool(p))

    def no_split(self):
        warn("`no_split` is deprecated, please use `split_none`.")
        return self.split_none()

    def split_none(self):
        """
        "Don't split the data and create an empty validation set."
        
        ----why:
        `ItemList.split_none`:
            1. don't split `self`, just create an empty `ItemList` for validation set 
            2. create and return an `ItemLists` object with 
               with `self` and empty validation `ItemList`.
        """
        # create an empty itemlist
        val = self[[]]
        # make sure `ignore_empty=True`
        val.ignore_empty = True
        # create an itemlists with `self` and `val`
        return self._split(self.path, self, val)

    def split_by_list(self, train, valid):
        """
        "Split the data between `train` and `valid`."

         `ItemList.split_by_list`:
            1. it actually uses `_split` to provide `ItemLists`
            2. to instantiate an `ItemLists` object with two `ItemList` attached
            3. this `ItemLists` is what `ItemList.split_by_folder`,
               and `split_by_idxs` return.
               
        ----inputs: 
            1. instead of taking in two lists above,
            2. it takes two `self[train_idx]` and `self[valid_idx]`
            3. these are two new `ItemList`s on training and validation sets
        
        ----internals
        `ItemLists.__init__`:
            1. it takes 3 inputs
            2. `path`: path_data
            3. `train`: new `ItemList` for training set
            4. `valid`: new `ItemList` for validation set
        """
        return self._split(self.path, train, valid)

    def split_by_idxs(self, train_idx, valid_idx):
        """
        "Split the data between `train_idx` and `valid_idx`."
        
        `ItemList.split_by_idxs`:
            1. `train_idx`: a list of indxs for training set
            2. `valid_idx`: a list of indxs for validation set
            3. does the same job as `ItemList.split_by_folder`

        ----internals
        `ItemList.split_by_list`: 
            1. instead of taking in two lists above,
            2. it takes two `self[train_idx]` and `self[valid_idx]`
            3. these are two new `ItemList`s on training and validation sets
        """
        return self.split_by_list(self[train_idx], self[valid_idx])

    def split_by_idx(self, valid_idx:Collection[int])->'ItemLists':
        "Split the data according to the indexes in `valid_idx`."
        #train_idx = [i for i in range_of(self.items) if i not in valid_idx]
        train_idx = np.setdiff1d(arange_of(self.items), valid_idx)
        return self.split_by_idxs(train_idx, valid_idx)

    def _get_by_folder(self, name):
        """
        `ItemList._get_by_folder`:
            1. files of all folders are in the same list of `self.items`;
            2. index all the files with `for i in range_of(self)`
            3. choose a subfolder `name`
            4. select all files with the `name` on into a list
            5. return the list

        ----internals:
        `range_of`: create a range from 0 to len(x) 
        """
        return [i for i in range_of(self) if (self.items[i].parts[self.num_parts] if isinstance(self.items[i], Path)
                else self.items[i].split(os.path.sep)[0]) == name ]

    def split_by_folder(self, train:str='train', valid:str='valid')->'ItemLists':
        """
        "Split the data depending on the folder (`train` or `valid`) in which the filenames are."
        ----why:
        `ItemList.split_by_folder`:
            1. split a large ItemList into a training ItemList 
               and a validation ItemList;
            2. create an `ItemLists` and  attach both `ItemList` into it.

        ----inputs:
        `train`: the foldername for training set of files
        `valid`: the foldername for validation set of files
        
        ----internals
        `split_by_idxs`: does same job above using two group of idxs
        `_get_by_folder`: get all the idxs of files of a particular folder
        """
        return self.split_by_idxs(self._get_by_folder(train), self._get_by_folder(valid))

    def random_split_by_pct(self, valid_pct:float=0.2, seed:int=None):
        warn("`random_split_by_pct` is deprecated, please use `split_by_rand_pct`.")
        return self.split_by_rand_pct(valid_pct=valid_pct, seed=seed)

    def split_by_rand_pct(self, valid_pct:float=0.2, seed:int=None)->'ItemLists':
        "Split the items randomly by putting `valid_pct` in the validation set, optional `seed` can be passed."
        if valid_pct==0.: return self.split_none()
        if seed is not None: np.random.seed(seed)
        rand_idx = np.random.permutation(range_of(self))
        cut = int(valid_pct * len(self))
        return self.split_by_idx(rand_idx[:cut])

    def split_subsets(self, train_size:float, valid_size:float, seed=None) -> 'ItemLists':
        "Split the items into train set with size `train_size * n` and valid set with size `valid_size * n`."
        assert 0 < train_size < 1
        assert 0 < valid_size < 1
        assert train_size + valid_size <= 1.
        if seed is not None: np.random.seed(seed)
        n = len(self.items)
        rand_idx = np.random.permutation(range(n))
        train_cut, valid_cut = int(train_size * n), int(valid_size * n)
        return self.split_by_idxs(rand_idx[:train_cut], rand_idx[-valid_cut:])

    def split_by_valid_func(self, func:Callable)->'ItemLists':
        "Split the data by result of `func` (which returns `True` for validation set)."
        valid_idx = [i for i,o in enumerate(self.items) if func(o)]
        return self.split_by_idx(valid_idx)

    def split_by_files(self, valid_names:'ItemList')->'ItemLists':
        "Split the data by using the names in `valid_names` for validation."
        if isinstance(self.items[0], Path): return self.split_by_valid_func(lambda o: o.name in valid_names)
        else: return self.split_by_valid_func(lambda o: os.path.basename(o) in valid_names)

    def split_by_fname_file(self, fname:PathOrStr, path:PathOrStr=None)->'ItemLists':
        "Split the data by using the names in `fname` for the validation set. `path` will override `self.path`."
        path = Path(ifnone(path, self.path))
        valid_names = loadtxt_str(path/fname)
        return self.split_by_files(valid_names)

    def split_from_df(self, col:IntsOrStrs=2):
        "Split the data from the `col` in the dataframe in `self.inner_df`."
        valid_idx = np.where(self.inner_df.iloc[:,df_names_to_idx(col, self.inner_df)])[0]
        return self.split_by_idx(valid_idx)

    def get_label_cls(self, labels, label_cls:Callable=None, label_delim:str=None, **kwargs):
        """
        "Return `label_cls` or guess one from the first element of `labels`."
        
        `ItemList.get_label_cls`: 
            1. get `label_cls` from `labels`
            2. like a look-up or match-up function
            3. if `label_cls` or `self.label_cls` is available, return `label_cls`
            4. if `label_delim` is available, return `MultiCategoryList`
            5. if first item of `labels` is float, return `FloatList`
            6. if first item of `labels` is int, return `CategoryList`
            7. if first item of `labels` is collection, return `MultiCategoryList`
            8. otherwise, return `ItemList`
        """
        if label_cls is not None:               return label_cls
        if self.label_cls is not None:          return self.label_cls
        if label_delim is not None:             return MultiCategoryList
        it = index_row(labels,0)
        if isinstance(it, (float, np.float32)): return FloatList
        if isinstance(try_int(it), (str, Integral)):  return CategoryList
        if isinstance(it, Collection):          return MultiCategoryList
        return ItemList #self.__class__

    def _label_from_list(self, labels:Iterator, label_cls:Callable=None, from_item_lists:bool=False, **kwargs)->'LabelList':
        """
        "Label `self.items` with `labels`."

        ----what and how
        `ItemList._label_from_list`:
            1. make sure `from_item_lists` set `True`, 
               otherwise exception message
            2. turn `labels` into an array of `labels`
            3. get `label_cls` out of `labels`
            4. instantiate `label_cls` with `labels`, `path`, `kwargs` to `y`
            5. create a `LabelList` object with `self` and `y`
            6. return this labellists 
        
        ----internals
        `ItemList.get_label_cls`: get `label_cls` from `labels`
        `label_cls(labels, path=self.path, **kwargs)`: instantiate a label cls
        `self._label_list(x=self, y=y)`: instantiate a `LabelList`

        ----Note
        `from_item_lists`:
            1. set `True` and 
            2. introduced since and by `ItemLists.__getattr__`
            3. passing down here through `kwargs`
        """
        if not from_item_lists:
            raise Exception("Your data isn't split, if you don't want a validation set, please use `split_none`.")
        labels = array(labels, dtype=object)
        label_cls = self.get_label_cls(labels, label_cls=label_cls, **kwargs)
        y = label_cls(labels, path=self.path, **kwargs)
        res = self._label_list(x=self, y=y)
        return res

    def label_from_df(self, cols:IntsOrStrs=1, label_cls:Callable=None, **kwargs):
        "Label `self.items` from the values in `cols` in `self.inner_df`."
        labels = self.inner_df.iloc[:,df_names_to_idx(cols, self.inner_df)]
        assert labels.isna().sum().sum() == 0, f"You have NaN values in column(s) {cols} of your dataframe, please fix it."
        if is_listy(cols) and len(cols) > 1 and (label_cls is None or label_cls == MultiCategoryList):
            new_kwargs,label_cls = dict(one_hot=True, classes= cols),MultiCategoryList
            kwargs = {**new_kwargs, **kwargs}
        return self._label_from_list(_maybe_squeeze(labels), label_cls=label_cls, **kwargs)

    def label_const(self, const:Any=0, label_cls:Callable=None, **kwargs)->'LabelList':
        "Label every item with `const`."
        return self.label_from_func(func=lambda o: const, label_cls=label_cls, **kwargs)

    def label_empty(self, **kwargs):
        "Label every item with an `EmptyLabel`."
        kwargs['label_cls'] = EmptyLabelList
        return self.label_from_func(func=lambda o: 0., **kwargs)

    def label_from_func(self, func:Callable, label_cls:Callable=None, **kwargs)->'LabelList':
        """
        "Apply `func` to every input to get its label."
        
        `ItemList.label_from_func`: 
            1. apply `func` to every item of `self.items`
            2. put all the output of func into a list
            3. pass the list and `label_cls` and `**kwargs` to 
            4. `ItemList._label_from_list` to create labels
            
        """
        return self._label_from_list([func(o) for o in self.items], label_cls=label_cls, **kwargs)

    def label_from_folder(self, label_cls:Callable=None, **kwargs)->'LabelList':
        """
        "Give a label to each filename depending on its folder."
        ----why
        `ItemList.label_from_folder`:
            1. build a lambda func to extract subfoldername from a file
            2. run `ItemList.label_from_func` with inputs like `func`,
               `label_cls` and `**kwargs`
            3. to create labels to dataset
        """
        return self.label_from_func(func=lambda o: (o.parts if isinstance(o, Path) else o.split(os.path.sep))[-2],
                                    label_cls=label_cls, **kwargs)

    def label_from_re(self, pat:str, full_path:bool=False, label_cls:Callable=None, **kwargs)->'LabelList':
        "Apply the re in `pat` to determine the label of every filename.  If `full_path`, search in the full name."
        pat = re.compile(pat)
        def _inner(o):
            s = str((os.path.join(self.path,o) if full_path else o).as_posix())
            res = pat.search(s)
            assert res,f'Failed to find "{pat}" in "{s}"'
            return res.group(1)
        return self.label_from_func(_inner, label_cls=label_cls, **kwargs)

    def databunch(self, **kwargs):
        "To throw a clear error message when the data wasn't split and labeled."
        raise Exception("Your data is neither split nor labeled, can't turn it into a `DataBunch` yet.")

class EmptyLabelList(ItemList):
    "Basic `ItemList` for dummy labels."
    def get(self, i): return EmptyLabel()
    def reconstruct(self, t:Tensor, x:Tensor=None):
        if len(t.size()) == 0: return EmptyLabel()
        return self.x.reconstruct(t,x) if has_arg(self.x.reconstruct, 'x') else self.x.reconstruct(t)

class CategoryProcessor(PreProcessor):
    "`PreProcessor` that create `classes` from `ds.items` and handle the mapping."
    def __init__(self, ds:ItemList):
        self.create_classes(ds.classes)
        self.state_attrs,self.warns = ['classes'],[]

    def create_classes(self, classes):
        self.classes = classes
        if classes is not None: self.c2i = {v:k for k,v in enumerate(classes)}
        
        """
        How does `CategoryProcessor.create_classes` work?
        `classes`: a list of unique and sorted labels
        `self`: an object of `CateogryProcessor`
        `c2i`: a dictionary whose keys are `classes` and values are indexes starting to increase from 0
        It assigns `classes` to `self.classes` and creates `self.c2i` with `classes` as keys.
        """

    def generate_classes(self, items):
        "Generate classes from `items` by taking the sorted unique values."
        return uniqueify(items, sort=True)

    def process_one(self,item):
        """
        item: any element of items
        c2i: a dictionary from classes to indexes
        it basically uses `item` as key of `c2i` to return `index` as the value 
        """
        if isinstance(item, EmptyLabel): return item
        res = self.c2i.get(item,None)
        if res is None: self.warns.append(str(item))
        return res

    def process(self, ds):
        if self.classes is None: self.create_classes(self.generate_classes(ds.items))
        ds.classes = self.classes
        ds.c2i = self.c2i
        super().process(ds)
        """
        How does `CategoryProcessor.process` work?
        `self` is an object of `CategoryProcessor`
        `ds` is an object of `CategoryList`
        `classes` is a list of unique labels
        `c2i` is a dictionary from `classes` to `indexes`
        it basically generates values for `ds.classes` and `ds.c2i` which previously are `None` and non-exist respectively.
        """

    def __getstate__(self): return {n:getattr(self,n) for n in self.state_attrs}
    def __setstate__(self, state:dict):
        self.create_classes(state['classes'])
        self.state_attrs = state.keys()
        for n in state.keys():
            if n!='classes': setattr(self, n, state[n])

class CategoryListBase(ItemList):
    "Basic `ItemList` for classification."
    def __init__(self, items:Iterator, classes:Collection=None, **kwargs):
        """
        ----what
        `CategoryListBase.__init__`:
            1. create a categorylistbase for labels
            2. add unique properties beyond `ItemList`: 
               `classes`, `filter_missing_y`
            2. it is to be subclasses for different kinds of labels situations

        ----internals
            1. assign default arg `classes` to property `classes`
            2. set property `filter_missing_y` `True`
            3. inherit from an ItemList 
            4. add `classes` to `copy_new`
        """
        self.classes=classes
        self.filter_missing_y = True
        super().__init__(items, **kwargs)
        self.copy_new.append('classes')

    @property
    def c(self): return len(self.classes)

class CategoryList(CategoryListBase):
    "Basic `ItemList` for single classification labels."
    _processor=CategoryProcessor
    def __init__(self, items:Iterator, classes:Collection=None, label_delim:str=None, **kwargs):
        """
        `CategoryList.__init__`:
            1. create a categorylist for labels

        ----internals
            1. non-default args: `items`/`labels`
            2. default args: `classes` and `label_delim`
            2. instantiate `CategoryListBase`
            3. add `CrossEntropyFlat()` as its `loss_func`

        ----Note
            1. when passing `path=self.path` onto `CategoryList.__init__`
            2. it has to enter into `kwargs`, as arbitrary args
            3. as `path=self.path` is not positional, not default args
        """
        super().__init__(items, classes=classes, **kwargs)
        self.loss_func = CrossEntropyFlat()

    def get(self, i):
        o = self.items[i]
        if o is None: return None
        return Category(o, self.classes[o])

    def analyze_pred(self, pred, thresh:float=0.5): return pred.argmax()

    def reconstruct(self, t):
        return Category(t, self.classes[t])

class MultiCategoryProcessor(CategoryProcessor):
    "`PreProcessor` that create `classes` from `ds.items` and handle the mapping."
    def __init__(self, ds:ItemList, one_hot:bool=False):
        super().__init__(ds)
        self.one_hot = one_hot
        self.state_attrs.append('one_hot')

    def process_one(self,item):
        if self.one_hot or isinstance(item, EmptyLabel): return item
        res = [super(MultiCategoryProcessor, self).process_one(o) for o in item]
        return [r for r in res if r is not None]

    def generate_classes(self, items):
        "Generate classes from `items` by taking the sorted unique values."
        classes = set()
        for c in items: classes = classes.union(set(c))
        classes = list(classes)
        classes.sort()
        return classes

class MultiCategoryList(CategoryListBase):
    "Basic `ItemList` for multi-classification labels."
    _processor=MultiCategoryProcessor
    def __init__(self, items:Iterator, classes:Collection=None, label_delim:str=None, one_hot:bool=False, **kwargs):
        if label_delim is not None: items = array(csv.reader(items.astype(str), delimiter=label_delim))
        super().__init__(items, classes=classes, **kwargs)
        if one_hot:
            assert classes is not None, "Please provide class names with `classes=...`"
            self.processor = [MultiCategoryProcessor(self, one_hot=True)]
        self.loss_func = BCEWithLogitsFlat()
        self.one_hot = one_hot
        self.copy_new += ['one_hot']

    def get(self, i):
        o = self.items[i]
        if o is None: return None
        if self.one_hot: return self.reconstruct(o.astype(np.float32))
        return MultiCategory(one_hot(o, self.c), [self.classes[p] for p in o], o)

    def analyze_pred(self, pred, thresh:float=0.5):
        return (pred >= thresh).float()

    def reconstruct(self, t):
        o = [i for i in range(self.c) if t[i] == 1.]
        return MultiCategory(t, [self.classes[p] for p in o], o)

class FloatList(ItemList):
    "`ItemList` suitable for storing the floats in items for regression. Will add a `log` if this flag is `True`."
    def __init__(self, items:Iterator, log:bool=False, classes:Collection=None, **kwargs):
        super().__init__(np.array(items, dtype=np.float32), **kwargs)
        self.log = log
        self.copy_new.append('log')
        self.c = self.items.shape[1] if len(self.items.shape) > 1 else 1
        self.loss_func = MSELossFlat()

    def get(self, i):
        o = super().get(i)
        return FloatItem(np.log(o) if self.log else o)

    def reconstruct(self,t): return FloatItem(t.numpy())

class ItemLists():
    "An `ItemList` for each of `train` and `valid` (optional `test`)."
    def __init__(self, path:PathOrStr, train:ItemList, valid:ItemList):
        self.path,self.train,self.valid,self.test = Path(path),train,valid,None
        if not self.train.ignore_empty and len(self.train.items) == 0:
            warn("Your training set is empty. If this is by design, pass `ignore_empty=True` to remove this warning.")
        if not self.valid.ignore_empty and len(self.valid.items) == 0:
            warn("""Your validation set is empty. If this is by design, use `split_none()`
                 or pass `ignore_empty=True` when labelling to remove this warning.""")
        if isinstance(self.train, LabelList): self.__class__ = LabelLists

    def __dir__(self)->List[str]:
        default_dir = dir(type(self)) + list(self.__dict__.keys())
        add_ons = ['label_const', 'label_empty', 'label_from_df', 'label_from_folder', 'label_from_func',
                   'label_from_list', 'label_from_re']
        return default_dir + add_ons

    def __repr__(self)->str:
        return f'{self.__class__.__name__};\n\nTrain: {self.train};\n\nValid: {self.valid};\n\nTest: {self.test}'

    def __getattr__(self, k):
        """
        `ItemLists.__getattr__`:
            1. apply `ItemList.k` to `self.train` and `self.valid`
            2. turn `ItemLists` to `LabelLists`
            3. call `LabelLists.process`

        ----inputs
        `k`: string, a method name of `ItemList`

        ----internals
        `LabelLists.process`: to process both labellists inside

        """
        # use built-in `getattr` to get method `ItemList.k` 
        # on `self.train` into an object `ft`
        ft = getattr(self.train, k)
        if not isinstance(ft, Callable): return ft
        # assign method `self.valid.k` to `fv`
        fv = getattr(self.valid, k)
        # make sure `ft`, `fv` are callable methods
        assert isinstance(fv, Callable)
        # `args` returns position arguments of current frame/method        
        def _inner(*args, **kwargs):
            # run `self.train.label_from_folder` and  
            # passing `from_item_lists=True` onto its own internal func
            self.train = ft(*args, from_item_lists=True, **kwargs)
            # make sure self.train is a LabelList now
            assert isinstance(self.train, LabelList)
            # add {'label_cls`:self.train.y.__class__} to kwargs
            kwargs['label_cls'] = self.train.y.__class__
            # run `self.valid.label_from_folder` like above
            self.valid = fv(*args, from_item_lists=True, **kwargs)
            # turn self from itemlists into labellists
            self.__class__ = LabelLists
            # let's this labellists processes itself
            self.process()
            return self
        return _inner

    def __setstate__(self,data:Any): self.__dict__.update(data)

    @property
    def lists(self):
        res = [self.train,self.valid]
        if self.test is not None: res.append(self.test)
        return res

    def label_from_lists(self, train_labels:Iterator, valid_labels:Iterator, label_cls:Callable=None, **kwargs)->'LabelList':
        "Use the labels in `train_labels` and `valid_labels` to label the data. `label_cls` will overwrite the default."
        label_cls = self.train.get_label_cls(train_labels, label_cls)
        self.train = self.train._label_list(x=self.train, y=label_cls(train_labels, **kwargs))
        self.valid = self.valid._label_list(x=self.valid, y=self.train.y.new(valid_labels, **kwargs))
        self.__class__ = LabelLists
        self.process()
        return self

    def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
        "Set `tfms` to be applied to the xs of the train and validation set."
        if not tfms: tfms=(None,None)
        assert is_listy(tfms) and len(tfms) == 2, "Please pass a list of two lists of transforms (train and valid)."
        self.train.transform(tfms[0], **kwargs)
        self.valid.transform(tfms[1], **kwargs)
        if self.test: self.test.transform(tfms[1], **kwargs)
        return self

    def transform_y(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
        "Set `tfms` to be applied to the ys of the train and validation set."
        if not tfms: tfms=(None,None)
        self.train.transform_y(tfms[0], **kwargs)
        self.valid.transform_y(tfms[1], **kwargs)
        if self.test: self.test.transform_y(tfms[1], **kwargs)
        return self

    def databunch(self, **kwargs):
        "To throw a clear error message when the data wasn't labeled."
        raise Exception("Your data isn't labeled, can't turn it into a `DataBunch` yet!")

class LabelLists(ItemLists):
    "A `LabelList` for each of `train` and `valid` (optional `test`)."
    def get_processors(self):
        "Read the default class processors if none have been set."
        procs_x,procs_y = listify(self.train.x._processor),listify(self.train.y._processor)
        xp = ifnone(self.train.x.processor, [p(ds=self.train.x) for p in procs_x])
        yp = ifnone(self.train.y.processor, [p(ds=self.train.y) for p in procs_y])
        return xp,yp

    def process(self):
        "Process the inner datasets."
        xp,yp = self.get_processors()
        for ds,n in zip(self.lists, ['train','valid','test']): ds.process(xp, yp, name=n)
        #progress_bar clear the outputs so in some case warnings issued during processing disappear.
        for ds in self.lists:
            if getattr(ds, 'warn', False): warn(ds.warn)
        return self

    def filter_by_func(self, func:Callable):
        for ds in self.lists: ds.filter_by_func(func)
        return self

    def databunch(self, path:PathOrStr=None, bs:int=64, val_bs:int=None, num_workers:int=defaults.cpus,
                  dl_tfms:Optional[Collection[Callable]]=None, device:torch.device=None, collate_fn:Callable=data_collate,
                  no_check:bool=False, **kwargs)->'DataBunch':
        "Create an `DataBunch` from self, `path` will override `self.path`, `kwargs` are passed to `DataBunch.create`."
        path = Path(ifnone(path, self.path))
        data = self.x._bunch.create(self.train, self.valid, test_ds=self.test, path=path, bs=bs, val_bs=val_bs,
                                    num_workers=num_workers, device=device, collate_fn=collate_fn, no_check=no_check, **kwargs)
        if getattr(self, 'normalize', False):#In case a normalization was serialized
            norm = self.normalize
            data.normalize((norm['mean'], norm['std']), do_x=norm['do_x'], do_y=norm['do_y'])
        data.label_list = self
        return data

    def add_test(self, items:Iterator, label:Any=None):
        "Add test set containing `items` with an arbitrary `label`."
        # if no label passed, use label of first training item
        if label is None: labels = EmptyLabelList([0] * len(items))
        else: labels = self.valid.y.new([label] * len(items)).process()
        if isinstance(items, ItemList): items = self.valid.x.new(items.items, inner_df=items.inner_df).process()
        else: items = self.valid.x.new(items).process()
        self.test = self.valid.new(items, labels)
        return self

    def add_test_folder(self, test_folder:str='test', label:Any=None):
        "Add test set containing items from `test_folder` and an arbitrary `label`."
        # note: labels will be ignored if available in the test dataset
        items = self.x.__class__.from_folder(self.path/test_folder)
        return self.add_test(items.items, label=label)

    @classmethod
    def load_state(cls, path:PathOrStr, state:dict):
        "Create a `LabelLists` with empty sets from the serialized `state`."
        path = Path(path)
        train_ds = LabelList.load_state(path, state)
        valid_ds = LabelList.load_state(path, state)
        return LabelLists(path, train=train_ds, valid=valid_ds)

    @classmethod
    def load_empty(cls, path:PathOrStr, fn:PathOrStr='export.pkl'):
        "Create a `LabelLists` with empty sets from the serialized file in `path/fn`."
        path = Path(path)
        state = torch.load(open(path/fn, 'rb'))
        return LabelLists.load_state(path, state)

def _check_kwargs(ds:ItemList, tfms:TfmList, **kwargs):
    tfms = listify(tfms)
    if (tfms is None or len(tfms) == 0) and len(kwargs) == 0: return
    if len(ds.items) >= 1:
        x = ds[0]
        try: x.apply_tfms(tfms, **kwargs)
        except Exception as e:
            raise Exception(f"It's not possible to apply those transforms to your dataset:\n {e}")

class LabelList(Dataset):
    "A list of inputs `x` and labels `y` with optional `tfms`."
    def __init__(self, x:ItemList, y:ItemList, tfms:TfmList=None, tfm_y:bool=False, **kwargs):
        """
        ----what
        `LabelList.__init__`:
            1. instantiate a labellist to gather 
            2. inputs `x` and labels `y`
            3. together with `tfms`
            4. `LabelList` inherits from `torch.utils.data.Dataset`
            5. source `Dataset` for details

        ----internals
            1. assign non-default args `x`, `y` to properties `x`, `y`
            2. assign default args `tfm_y` to properties `tfm_y`
            3. assign property `item` to None
            4. do `self.transforms` with `tfms` (default args ) and `**kwargs` 
        """
        self.x,self.y,self.tfm_y = x,y,tfm_y
        self.y.x = x
        self.item=None
        self.transform(tfms, **kwargs)

    def __len__(self)->int: return len(self.x) if self.item is None else 1

    @contextmanager
    def set_item(self,item):
        "For inference, will briefly replace the dataset with one that only contains `item`."
        self.item = self.x.process_one(item)
        yield None
        self.item = None

    def __repr__(self)->str:
        items = [self[i] for i in range(min(5,len(self.items)))]
        res = f'{self.__class__.__name__} ({len(self.items)} items)\n'
        res += f'x: {self.x.__class__.__name__}\n{show_some([i[0] for i in items])}\n'
        res += f'y: {self.y.__class__.__name__}\n{show_some([i[1] for i in items])}\n'
        return res + f'Path: {self.path}'

    def predict(self, res):
        "Delegates predict call on `res` to `self.y`."
        return self.y.predict(res)

    @property
    def c(self): return self.y.c

    def new(self, x, y, **kwargs)->'LabelList':
        if isinstance(x, ItemList):
            return self.__class__(x, y, tfms=self.tfms, tfm_y=self.tfm_y, **self.tfmargs)
        else:
            return self.new(self.x.new(x, **kwargs), self.y.new(y, **kwargs)).process()

    def __getattr__(self,k:str)->Any:
        x = super().__getattribute__('x')
        res = getattr(x, k, None)
        if res is not None and k not in ['classes', 'c']: return res
        y = super().__getattribute__('y')
        res = getattr(y, k, None)
        if res is not None: return res
        raise AttributeError(k)

    def __setstate__(self,data:Any): self.__dict__.update(data)

    def __getitem__(self,idxs:Union[int,np.ndarray])->'LabelList':
        idxs = try_int(idxs)
        if isinstance(idxs, Integral):
            if self.item is None: x,y = self.x[idxs],self.y[idxs]
            else:                 x,y = self.item   ,0
            if self.tfms or self.tfmargs:
                x = x.apply_tfms(self.tfms, **self.tfmargs)
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve':False})
            if y is None: y=0
            return x,y
        else: return self.new(self.x[idxs], self.y[idxs])

    def to_df(self)->None:
        "Create `pd.DataFrame` containing `items` from `self.x` and `self.y`."
        return pd.DataFrame(dict(x=self.x._relative_item_paths(), y=[str(o) for o in self.y]))

    def to_csv(self, dest:str)->None:
        "Save `self.to_df()` to a CSV file in `self.path`/`dest`."
        self.to_df().to_csv(self.path/dest, index=False)

    def get_state(self, **kwargs):
        "Return the minimal state for export."
        state = {'x_cls':self.x.__class__, 'x_proc':self.x.processor,
                 'y_cls':self.y.__class__, 'y_proc':self.y.processor,
                 'tfms':self.tfms, 'tfm_y':self.tfm_y, 'tfmargs':self.tfmargs}
        if hasattr(self, 'tfms_y'):    state['tfms_y']    = self.tfms_y
        if hasattr(self, 'tfmargs_y'): state['tfmargs_y'] = self.tfmargs_y
        return {**state, **kwargs}

    def export(self, fn:PathOrStr, **kwargs):
        "Export the minimal state and save it in `fn` to load an empty version for inference."
        pickle.dump(self.get_state(**kwargs), open(fn, 'wb'))

    @classmethod
    def load_empty(cls, path:PathOrStr, fn:PathOrStr):
        "Load the state in `fn` to create an empty `LabelList` for inference."
        return cls.load_state(path, pickle.load(open(Path(path)/fn, 'rb')))

    @classmethod
    def load_state(cls, path:PathOrStr, state:dict) -> 'LabelList':
        "Create a `LabelList` from `state`."
        x = state['x_cls']([], path=path, processor=state['x_proc'], ignore_empty=True)
        y = state['y_cls']([], path=path, processor=state['y_proc'], ignore_empty=True)
        res = cls(x, y, tfms=state['tfms'], tfm_y=state['tfm_y'], **state['tfmargs']).process()
        if state.get('tfms_y', False):    res.tfms_y    = state['tfms_y']
        if state.get('tfmargs_y', False): res.tfmargs_y = state['tfmargs_y']
        if state.get('normalize', False): res.normalize = state['normalize']
        return res

    def process(self, xp:PreProcessor=None, yp:PreProcessor=None, name:str=None):
        "Launch the processing on `self.x` and `self.y` with `xp` and `yp`."
        self.y.process(yp)
        if getattr(self.y, 'filter_missing_y', False):
            filt = array([o is None for o in self.y.items])
            if filt.sum()>0:
                #Warnings are given later since progress_bar might make them disappear.
                self.warn = f"You are labelling your items with {self.y.__class__.__name__}.\n"
                self.warn += f"Your {name} set contained the following unknown labels, the corresponding items have been discarded.\n"
                for p in self.y.processor:
                    if len(getattr(p, 'warns', [])) > 0:
                        warnings = list(set(p.warns))
                        self.warn += ', '.join(warnings[:5])
                        if len(warnings) > 5: self.warn += "..."
                    p.warns = []
                self.x,self.y = self.x[~filt],self.y[~filt]
        self.x.process(xp)
        return self

    def filter_by_func(self, func:Callable):
        filt = array([func(x,y) for x,y in zip(self.x.items, self.y.items)])
        self.x,self.y = self.x[~filt],self.y[~filt]
        return self

    def transform(self, tfms:TfmList, tfm_y:bool=None, **kwargs):
        "Set the `tfms` and `tfm_y` value to be applied to the inputs and targets."
        _check_kwargs(self.x, tfms, **kwargs)
        if tfm_y is None: tfm_y = self.tfm_y
        if tfm_y: _check_kwargs(self.y, tfms, **kwargs)
        self.tfms,  self.tfmargs   = tfms,kwargs
        self.tfm_y, self.tfmargs_y = tfm_y,kwargs
        self.tfms_y = None if tfms is None else list(filter(lambda t: t.use_on_y, listify(tfms)))
        return self

    def transform_y(self, tfms:TfmList=None, **kwargs):
        "Set `tfms` to be applied to the targets only."
        _check_kwargs(self.y, tfms, **kwargs)
        self.tfm_y=True
        if tfms is None:
            self.tfms_y = list(filter(lambda t: t.use_on_y, listify(self.tfms)))
            self.tfmargs_y = {**self.tfmargs, **kwargs}
        else:
            tfms = list(filter(lambda t: t.use_on_y, tfms))
            self.tfms_y,self.tfmargs_y = tfms,kwargs
        return self

    def databunch(self, **kwargs):
        "To throw a clear error message when the data wasn't split."
        raise Exception("Your data isn't split, if you don't want a validation set, please use `split_none`")

@classmethod
def _databunch_load_empty(cls, path, fname:str='export.pkl'):
    "Load an empty `DataBunch` from the exported file in `path/fname` with optional `tfms`."
    sd = LabelLists.load_empty(path, fn=fname)
    return sd.databunch()

DataBunch.load_empty = _databunch_load_empty

class MixedProcessor(PreProcessor):
    def __init__(self, procs:Collection[Union[PreProcessor, Collection[PreProcessor]]]):
        self.procs = procs

    def process_one(self, item:Any):
        res = []
        for procs, i in zip(self.procs, item):
            for p in procs: i = p.process_one(i)
            res.append(i)
        return res

    def process(self, ds:Collection):
        for procs, il in zip(self.procs, ds.item_lists):
            for p in procs: p.process(il)

class MixedItem(ItemBase):
    def __init__(self, items):
        self.obj = items
        self.data = [item.data for item in items]

    def __repr__(self): return '\n'.join([f'{self.__class__.__name__}'] + [repr(item) for item in self.obj])

    def apply_tfms(self, tfms:Collection, **kwargs):
        self.obj = [item.apply_tfms(t, **kwargs) for item,t in zip(self.obj, tfms)]
        self.data = [item.data for item in self.obj]
        return self

class MixedItemList(ItemList):

    def __init__(self, item_lists, path:PathOrStr=None, label_cls:Callable=None, inner_df:Any=None,
                 x:'ItemList'=None, ignore_empty:bool=False, processor=None):
        self.item_lists = item_lists
        if processor is None:
            default_procs = [[p(ds=il) for p in listify(il._processor)] for il in item_lists]
            processor = MixedProcessor([ifnone(il.processor, dp) for il,dp in zip(item_lists, default_procs)])
        items = range_of(item_lists[0]) if len(item_lists) >= 1 else []
        if path is None and len(item_lists) >= 1: path = item_lists[0].path
        super().__init__(items, processor=processor, path=path,
                         label_cls=label_cls, inner_df=inner_df, x=x, ignore_empty=ignore_empty)

    def new(self, item_lists, processor:PreProcessor=None, **kwargs)->'ItemList':
        "Create a new `ItemList` from `items`, keeping the same attributes."
        processor = ifnone(processor, self.processor)
        copy_d = {o:getattr(self,o) for o in self.copy_new}
        kwargs = {**copy_d, **kwargs}
        return self.__class__(item_lists, processor=processor, **kwargs)

    def get(self, i):
        return MixedItem([il.get(i) for il in self.item_lists])

    def __getitem__(self,idxs:int)->Any:
        idxs = try_int(idxs)
        if isinstance(idxs, Integral): return self.get(idxs)
        else:
            item_lists = [il.new(il.items[idxs], inner_df=index_row(il.inner_df, idxs)) for il in self.item_lists]
            return self.new(item_lists, inner_df=index_row(self.inner_df, idxs))
