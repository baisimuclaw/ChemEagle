import os
import torch
import re
from functools import lru_cache
import layoutparser as lp
import pdf2image
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from molnextr import MolNexTR
from rxnim import RxnIM, MolDetect
from chemiener import ChemNER
from .chemrxnextractor import ChemRxnExtractor
from .tableextractor import TableExtractor
from .utils import *

class ChemIEToolkit:
    def __init__(self, device=None, model_dir=None, offline=False):
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model_dir = (
            os.path.abspath(os.path.expanduser(model_dir)) if model_dir else None
        )
        self.offline = bool(offline)

        self._molnextr = None
        self._rxnim = None
        self._pdfparser = None
        self._moldet = None
        self._chemrxnextractor = None
        self._chemner = None
        self._coref = None
        self._ocr_model = None

    def _local_model_path(self, filename):
        if not self.model_dir:
            return None
        candidate = os.path.join(self.model_dir, filename)
        return candidate if os.path.exists(candidate) else None

    def _checkpoint(self, filename, repo_id):
        local = self._local_model_path(filename)
        if local:
            return local
        if self.offline:
            raise FileNotFoundError(
                f"Offline ChemEAGLE worker requires {filename!r} in {self.model_dir!r}"
            )
        return hf_hub_download(repo_id, filename)

    @property
    def ocr_model(self):
        if self._ocr_model is None:
            import easyocr

            kwargs = {"gpu": self.device.type == "cuda", "download_enabled": not self.offline}
            if self.model_dir:
                easyocr_dir = os.path.join(self.model_dir, "easyocr")
                if self.offline and not os.path.isdir(easyocr_dir):
                    raise FileNotFoundError(
                        "Offline EasyOCR requires an easyocr directory under "
                        f"{self.model_dir!r}"
                    )
                kwargs["model_storage_directory"] = easyocr_dir
            self._ocr_model = easyocr.Reader(["en"], **kwargs)
        return self._ocr_model

    @property
    def molnextr(self):
        if self._molnextr is None:
            self.init_molnextr()
        return self._molnextr

    @lru_cache(maxsize=None)
    def init_molnextr(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        if ckpt_path is None:
            ckpt_path = self._checkpoint(
                "molnextr.pth", "CYF200127/ChemEAGLEModel"
            )
        self._molnextr = MolNexTR(ckpt_path, device=self.device)
    

    @property
    def rxnim(self):
        if self._rxnim is None:
            self.init_rxnim()
        return self._rxnim

    @lru_cache(maxsize=None)
    def init_rxnim(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        if ckpt_path is None:
            ckpt_path = self._checkpoint("rxn.ckpt", "CYF200127/ChemEAGLEModel")
        self._rxnim = RxnIM(
            ckpt_path,
            device=self.device,
            molnextr_model=self.molnextr,
            ocr_model=self.ocr_model,
            offline=self.offline,
        )
    

    @property
    def pdfparser(self):
        if self._pdfparser is None:
            self.init_pdfparser()
        return self._pdfparser

    @lru_cache(maxsize=None)
    def init_pdfparser(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        config_path = "lp://efficientdet/PubLayNet/tf_efficientdet_d1"
        self._pdfparser = lp.AutoLayoutModel(config_path, model_path=ckpt_path, device=self.device.type)
    

    @property
    def moldet(self):
        if self._moldet is None:
            self.init_moldet()
        return self._moldet

    @lru_cache(maxsize=None)
    def init_moldet(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        if ckpt_path is None:
            ckpt_path = self._checkpoint(
                "moldet.ckpt", "CYF200127/ChemEAGLEModel"
            )
        self._moldet = MolDetect(
            ckpt_path,
            device=self.device,
            molnextr_model=self.molnextr,
            ocr_model=self.ocr_model,
            offline=self.offline,
        )
        

    @property
    def coref(self):
        if self._coref is None:
            self.init_coref()
        return self._coref

    @lru_cache(maxsize=None)
    def init_coref(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        if ckpt_path is None:
            ckpt_path = self._checkpoint(
                "corefdet.ckpt", "CYF200127/ChemEAGLEModel"
            )
        self._coref = MolDetect(
            ckpt_path,
            device=self.device,
            coref=True,
            molnextr_model=self.molnextr,
            ocr_model=self.ocr_model,
            offline=self.offline,
        )


    @property
    def chemrxnextractor(self):
        if self._chemrxnextractor is None:
            self.init_chemrxnextractor()
        return self._chemrxnextractor

    @lru_cache(maxsize=None)
    def init_chemrxnextractor(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        if ckpt_path is None:
            ckpt_path = self._local_model_path("chemrxnextractor")
        if ckpt_path is None and self.offline:
            raise FileNotFoundError(
                "Offline ChemEAGLE worker requires a chemrxnextractor directory "
                f"under {self.model_dir!r}"
            )
        if ckpt_path is None:
            ckpt_path = snapshot_download(
                repo_id="amberwang/chemrxnextractor-training-modules"
            )
        self._chemrxnextractor = ChemRxnExtractor("", None, ckpt_path, self.device.type)

    def extract_reactions_from_strings(self, strings):
        return self.chemrxnextractor.rxn_extractor.get_reactions(strings)


    @property
    def chemner(self):
        if self._chemner is None:
            self.init_chemner()
        return self._chemner

    @lru_cache(maxsize=None)
    def init_chemner(self, ckpt_path=None):
        """
        Set model to custom checkpoint
        Parameters:
            ckpt_path: path to checkpoint to use, if None then will use default
        """
        if ckpt_path is None:
            ckpt_path = self._checkpoint("ner.ckpt", "CYF200127/ChemEAGLEModel")
        base_model = self._local_model_path("biobert-large-cased")
        if self.offline and base_model is None:
            raise FileNotFoundError(
                "Offline ChemNER requires biobert-large-cased under "
                f"{self.model_dir!r}"
            )
        self._chemner = ChemNER(
            ckpt_path,
            device=self.device,
            cache_dir=os.path.join(self.model_dir, "huggingface") if self.model_dir else None,
            roberta_checkpoint=base_model,
        )

    
    @property
    def tableextractor(self):
        return TableExtractor()


    def extract_figures_from_pdf(self, pdf, num_pages=None, output_bbox=False, output_image=True):
        """
        Find and return all figures from a pdf page
        Parameters:
            pdf: path to pdf
            num_pages: process only first `num_pages` pages, if `None` then process all
            output_bbox: whether to output bounding boxes for each individual entry of a table
            output_image: whether to include PIL image for figures. default is True
        Returns:
            list of content in the following format
            [
                { # first figure
                    'title': str,
                    'figure': {
                        'image': PIL image or None,
                        'bbox': list in form [x1, y1, x2, y2],
                    }
                    'table': {
                        'bbox': list in form [x1, y1, x2, y2] or empty list,
                        'content': {
                            'columns': list of column headers,
                            'rows': list of list of row content,
                        } or None
                    }
                    'footnote': str or empty,
                    'page': int
                }
                # more figures
            ]
        """
        pages = pdf2image.convert_from_path(pdf, last_page=num_pages)

        table_ext = self.tableextractor
        table_ext.set_pdf_file(pdf)
        table_ext.set_output_image(output_image)

        table_ext.set_output_bbox(output_bbox)
        
        return table_ext.extract_all_tables_and_figures(pages, self.pdfparser, content='figures')

    def extract_tables_from_pdf(self, pdf, num_pages=None, output_bbox=False, output_image=True):
        """
        Find and return all tables from a pdf page
        Parameters:
            pdf: path to pdf
            num_pages: process only first `num_pages` pages, if `None` then process all
            output_bbox: whether to include bboxes for individual entries of the table
            output_image: whether to include PIL image for figures. default is True
        Returns:
            list of content in the following format
            [
                { # first table
                    'title': str,
                    'figure': {
                        'image': PIL image or None,
                        'bbox': list in form [x1, y1, x2, y2] or empty list,
                    }
                    'table': {
                        'bbox': list in form [x1, y1, x2, y2] or empty list,
                        'content': {
                            'columns': list of column headers,
                            'rows': list of list of row content,
                        }
                    }
                    'footnote': str or empty,
                    'page': int
                }
                # more tables
            ]
        """
        pages = pdf2image.convert_from_path(pdf, last_page=num_pages)

        table_ext = self.tableextractor
        table_ext.set_pdf_file(pdf)
        table_ext.set_output_image(output_image)

        table_ext.set_output_bbox(output_bbox)
        
        return table_ext.extract_all_tables_and_figures(pages, self.pdfparser, content='tables')

    def extract_molecules_from_figures_in_pdf(self, pdf, batch_size=16, num_pages=None):
        """
        Get all molecules and their information from a pdf
        Parameters:
            pdf: path to pdf, or byte file
            batch_size: batch size for inference in all models
            num_pages: process only first `num_pages` pages, if `None` then process all
        Returns:
            list of figures and corresponding molecule info in the following format
            [
                {   # first figure
                    'image': ndarray of the figure image,
                    'molecules': [
                        {   # first molecule
                            'bbox': tuple in the form (x1, y1, x2, y2),
                            'score': float,
                            'image': ndarray of cropped molecule image,
                            'smiles': str,
                            'molfile': str
                        },
                        # more molecules
                    ],
                    'page': int
                },
                # more figures
            ]
        """
        figures = self.extract_figures_from_pdf(pdf, num_pages=num_pages, output_bbox=True)
        images = [figure['figure']['image'] for figure in figures]
        results = self.extract_molecules_from_figures(images, batch_size=batch_size)
        for figure, result in zip(figures, results):
            result['page'] = figure['page']
        return results
    
    def extract_molecule_bboxes_from_figures(self, figures, batch_size=16):
        """
        Return bounding boxes of molecules in images
        Parameters:
            figures: list of PIL or ndarray images
            batch_size: batch size for inference
        Returns:
            list of results for each figure in the following format
            [
                [   # first figure
                    {   # first bounding box
                        'category': str,
                        'bbox': tuple in the form (x1, y1, x2, y2),
                        'category_id': int,
                        'score': float
                    },
                    # more bounding boxes
                ],
                # more figures
            ]
        """
        figures = [convert_to_pil(figure) for figure in figures]
        return self.moldet.predict_images(figures, batch_size=batch_size)

    def extract_molecules_from_figures(self, figures, batch_size=16):
        """
        Get all molecules and their information from list of figures
        Parameters:
            figures: list of PIL or ndarray images
            batch_size: batch size for inference
        Returns:
            list of results for each figure in the following format
            [
                {   # first figure
                    'image': ndarray of the figure image,
                    'molecules': [
                        {   # first molecule
                            'bbox': tuple in the form (x1, y1, x2, y2),
                            'score': float,
                            'image': ndarray of cropped molecule image,
                            'smiles': str,
                            'molfile': str
                        },
                        # more molecules
                    ],
                },
                # more figures
            ]
        """
        bboxes = self.extract_molecule_bboxes_from_figures(figures, batch_size=batch_size)
        figures = [convert_to_cv2(figure) for figure in figures]
        results, cropped_images, refs = clean_bbox_output(figures, bboxes)
        mol_info = self.molnextr.predict_images(cropped_images, batch_size=batch_size)
        for info, ref in zip(mol_info, refs):
            ref.update(info)
        return results

    def extract_molecule_corefs_from_figures_in_pdf(self, pdf, batch_size=16, num_pages=None, molnextr = True, ocr = True):
        """
        Get all molecule bboxes and corefs from figures in pdf
        Parameters:
            pdf: path to pdf, or byte file
            batch_size: batch size for inference in all models
            num_pages: process only first `num_pages` pages, if `None` then process all
        Returns:
            list of results for each figure in the following format:
            [
                {
                    'bboxes': [
                        {   # first bbox
                            'category': '[Sup]', 
                            'bbox': (0.0050025012506253125, 0.38273870663142223, 0.9934967483741871, 0.9450094869920168), 
                            'category_id': 4, 
                            'score': -0.07593922317028046
                        },
                        # More bounding boxes
                    ],
                    'corefs': [
                        [0, 1],  # molecule bbox index, identifier bbox index
                        [3, 4],
                        # More coref pairs
                    ],
                    'page': int
                },
                # More figures
            ]
        """
        figures = self.extract_figures_from_pdf(pdf, num_pages=num_pages, output_bbox=True)
        images = [figure['figure']['image'] for figure in figures]
        results = self.extract_molecule_corefs_from_figures(images, batch_size=batch_size, molnextr=molnextr, ocr=ocr)
        for figure, result in zip(figures, results):
            result['page'] = figure['page']
        return results

    def extract_molecule_corefs_from_figures(self, figures, batch_size=16, molnextr=True, ocr=True):
        """
        Get all molecule bboxes and corefs from list of figures
        Parameters:
            figures: list of PIL or ndarray images
            batch_size: batch size for inference
        Returns:
            list of results for each figure in the following format:
            [
                {
                    'bboxes': [
                        {   # first bbox
                            'category': '[Sup]', 
                            'bbox': (0.0050025012506253125, 0.38273870663142223, 0.9934967483741871, 0.9450094869920168), 
                            'category_id': 4, 
                            'score': -0.07593922317028046
                        },
                        # More bounding boxes
                    ],
                    'corefs': [
                        [0, 1],  # molecule bbox index, identifier bbox index
                        [3, 4],
                        # More coref pairs
                    ],
                },
                # More figures
            ]
        """
        figures = [convert_to_pil(figure) for figure in figures]
        return self.coref.predict_images(figures, batch_size=batch_size, coref=True, molnextr = molnextr, ocr = ocr)
    
    def extract_reactions_from_figures_in_pdf(self, pdf, batch_size=16, num_pages=None, molnextr=True, ocr=True):
        """
        Get reaction information from figures in pdf
        Parameters:
            pdf: path to pdf, or byte file
            batch_size: batch size for inference in all models
            num_pages: process only first `num_pages` pages, if `None` then process all
            molnextr: whether to predict and return smiles and molfile info
            ocr: whether to predict and return text of conditions
        Returns:
            list of figures and corresponding molecule info in the following format
            [
                {
                    'figure': PIL image
                    'reactions': [
                        {
                            'reactants': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                    'smiles': str,
                                    'molfile': str,
                                },
                                # more reactants
                            ],
                            'conditions': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                    'text': list of str,
                                },
                                # more conditions
                            ],
                            'products': [
                                # same structure as reactants
                            ]
                        },
                        # more reactions
                    ],
                    'page': int
                },
                # more figures
            ]
        """
        figures = self.extract_figures_from_pdf(pdf, num_pages=num_pages, output_bbox=True)
        images = [figure['figure']['image'] for figure in figures]
        results = self.extract_reactions_from_figures(images, batch_size=batch_size, molnextr=molnextr, ocr=ocr)
        for figure, result in zip(figures, results):
            result['page'] = figure['page']
        return results

    def extract_reactions_from_figures(self, figures, batch_size=16, molnextr=True, ocr=True):
        """
        Get reaction information from list of figures
        Parameters:
            figures: list of PIL or ndarray images
            batch_size: batch size for inference in all models
            molnextr: whether to predict and return smiles and molfile info
            ocr: whether to predict and return text of conditions
        Returns:
            list of figures and corresponding molecule info in the following format
            [
                {
                    'figure': PIL image
                    'reactions': [
                        {
                            'reactants': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                    'smiles': str,
                                    'molfile': str,
                                },
                                # more reactants
                            ],
                            'conditions': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                    'text': list of str,
                                },
                                # more conditions
                            ],
                            'products': [
                                # same structure as reactants
                            ]
                        },
                        # more reactions
                    ],
                },
                # more figures
            ]

        """
        pil_figures = [convert_to_pil(figure) for figure in figures]
        results = []
        reactions = self.rxnim.predict_images(pil_figures, batch_size=batch_size, molnextr=molnextr, ocr=ocr)
        for figure, rxn in zip(figures, reactions):
            data = {
                'figure': figure,
                'reactions': rxn,
                }
            results.append(data)
        return results

    def extract_molecules_from_text_in_pdf(self, pdf, batch_size=16, num_pages=None):
        """
        Get molecules in text of given pdf

        Parameters:
            pdf: path to pdf, or byte file
            batch_size: batch size for inference in all models
            num_pages: process only first `num_pages` pages, if `None` then process all
        Returns:
            list of sentences and found molecules in the following format
            [
                {
                    'molecules': [
                        { # first paragraph
                            'text': str,
                            'labels': [
                                (str, int, int), # tuple of label, range start (inclusive), range end (exclusive)
                                # more labels
                            ]
                        },
                        # more paragraphs
                    ]
                    'page': int
                },
                # more pages
            ]
        """
        self.chemrxnextractor.set_pdf_file(pdf)
        self.chemrxnextractor.set_pages(num_pages)
        text = self.chemrxnextractor.get_paragraphs_from_pdf(num_pages)
        result = []
        for data in text:
            model_inp = []
            for paragraph in data['paragraphs']:
                model_inp.append(' '.join(paragraph).replace('\n', ''))
            output = self.chemner.predict_strings(model_inp, batch_size=batch_size)
            to_add = {
                'molecules': [{
                    'text': t,
                    'labels': labels,
                    } for t, labels in zip(model_inp, output)],
                'page': data['page']
            }
            result.append(to_add)
        return result


    def extract_reactions_from_text_in_pdf(self, pdf, num_pages=None):
        """
        Get reaction information from text in pdf
        Parameters:
            pdf: path to pdf
            num_pages: process only first `num_pages` pages, if `None` then process all
        Returns:
            list of pages and corresponding reaction info in the following format
            [
                {
                    'page': page number
                    'reactions': [
                        {
                            'tokens': list of words in relevant sentence,
                            'reactions' : [
                                {
                                    # key, value pairs where key is the label and value is a tuple
                                    # or list of tuples of the form (tokens, start index, end index)
                                    # where indices are for the corresponding token list and start and end are inclusive
                                }
                                # more reactions
                            ]
                        }
                        # more reactions in other sentences
                    ]
                },
                # more pages
            ]
        """
        self.chemrxnextractor.set_pdf_file(pdf)
        self.chemrxnextractor.set_pages(num_pages)
        return self.chemrxnextractor.extract_reactions_from_text()

    def extract_reactions_from_text_in_pdf_combined(self, pdf, num_pages=None):
        """
        Get reaction information from text in pdf and combined with corefs from figures
        Parameters:
            pdf: path to pdf
            num_pages: process only first `num_pages` pages, if `None` then process all
        Returns:
            list of pages and corresponding reaction info in the following format
            [
                {
                    'page': page number
                    'reactions': [
                        {
                            'tokens': list of words in relevant sentence,
                            'reactions' : [
                                {
                                    # key, value pairs where key is the label and value is a tuple
                                    # or list of tuples of the form (tokens, start index, end index)
                                    # where indices are for the corresponding token list and start and end are inclusive
                                }
                                # more reactions
                            ]
                        }
                        # more reactions in other sentences
                    ]
                },
                # more pages
            ]
        """
        results = self.extract_reactions_from_text_in_pdf(pdf, num_pages=num_pages)
        results_coref = self.extract_molecule_corefs_from_figures_in_pdf(pdf, num_pages=num_pages)
        return associate_corefs(results, results_coref)

    def extract_reactions_from_figures_and_tables_in_pdf(self, pdf, num_pages=None, batch_size=16, molnextr=True, ocr=True):
        """
        Get reaction information from figures and combine with table information in pdf
        Parameters:
            pdf: path to pdf, or byte file
            batch_size: batch size for inference in all models
            num_pages: process only first `num_pages` pages, if `None` then process all
            molnextr: whether to predict and return smiles and molfile info
            ocr: whether to predict and return text of conditions
        Returns:
            list of figures and corresponding molecule info in the following format
            [
                {
                    'figure': PIL image
                    'reactions': [
                        {
                            'reactants': [
                                {
                                    'category': str,
                                    'bbox': tuple (x1,x2,y1,y2),
                                    'category_id': int,
                                    'smiles': str,
                                    'molfile': str,
                                },
                                # more reactants
                            ],
                            'conditions': [
                                {
                                    'category': str,
                                    'text': list of str,
                                },
                                # more conditions
                            ],
                            'products': [
                                # same structure as reactants
                            ]
                        },
                        # more reactions
                    ],
                    'page': int
                },
                # more figures
            ]
        """
        figures = self.extract_figures_from_pdf(pdf, num_pages=num_pages, output_bbox=True)
        images = [figure['figure']['image'] for figure in figures]
        results = self.extract_reactions_from_figures(images, batch_size=batch_size, molnextr=molnextr, ocr=ocr)
        results = process_tables(figures, results, self.molnextr, batch_size=batch_size)
        results_coref = self.extract_molecule_corefs_from_figures_in_pdf(pdf, num_pages=num_pages)
        results = replace_rgroups_in_figure(figures, results, results_coref, self.molnextr, batch_size=batch_size)
        results = expand_reactions_with_backout(results, results_coref, self.molnextr)
        return results

    def extract_reactions_from_pdf(self, pdf, num_pages=None, batch_size=16):
        """
        Returns:
            dictionary of reactions from multimodal sources
            {
                'figures': [
                    {
                        'figure': PIL image
                        'reactions': [
                            {
                                'reactants': [
                                    {
                                        'category': str,
                                        'bbox': tuple (x1,x2,y1,y2),
                                        'category_id': int,
                                        'smiles': str,
                                        'molfile': str,
                                    },
                                    # more reactants
                                ],
                                'conditions': [
                                    {
                                        'category': str,
                                        'text': list of str,
                                    },
                                    # more conditions
                                ],
                                'products': [
                                    # same structure as reactants
                                ]
                            },
                            # more reactions
                        ],
                        'page': int
                    },
                    # more figures
                ]
                'text': [
                    {
                        'page': page number
                        'reactions': [
                            {
                                'tokens': list of words in relevant sentence,
                                'reactions' : [
                                    {
                                        # key, value pairs where key is the label and value is a tuple
                                        # or list of tuples of the form (tokens, start index, end index)
                                        # where indices are for the corresponding token list and start and end are inclusive
                                    }
                                    # more reactions
                                ]
                            }
                            # more reactions in other sentences
                        ]
                    },
                    # more pages
                ]
            }

        """
        figures = self.extract_figures_from_pdf(pdf, num_pages=num_pages, output_bbox=True)
        images = [figure['figure']['image'] for figure in figures]
        results = self.extract_reactions_from_figures(images, batch_size=batch_size, molnextr=True, ocr=True)
        table_expanded_results = process_tables(figures, results, self.molnextr, batch_size=batch_size)
        text_results = self.extract_reactions_from_text_in_pdf(pdf, num_pages=num_pages)
        results_coref = self.extract_molecule_corefs_from_figures_in_pdf(pdf, num_pages=num_pages)
        figure_results = replace_rgroups_in_figure(figures, table_expanded_results, results_coref, self.molnextr, batch_size=batch_size)
        table_expanded_results = expand_reactions_with_backout(figure_results, results_coref, self.molnextr)
        coref_expanded_results = associate_corefs(text_results, results_coref)
        return {
            'figures': table_expanded_results,
            'text': coref_expanded_results,
        }

if __name__=="__main__":
    model = ChemIEToolkit()
