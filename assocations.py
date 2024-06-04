import numpy as np 
import seaborn as sns 
import tkinter as tk
from tkinter import filedialog, messagebox
from mlxtend.frequent_patterns import association_rules
from tkinter import ttk
import itertools
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
#author Nada Hassan 20201187 

class AprioriUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Apriori Algorithm")
        
        self.file_path = tk.StringVar(value="C:/Users/nada/Downloads/Assignment1/Bakery.csv")
        self.min_support_count = tk.IntVar(value=50)
        self.min_confidence = tk.IntVar(value=50)
        self.selected_percentage = tk.IntVar(value=100)
        
        self.create_widgets()
        
    def create_widgets(self):
        tk.Label(self.root, text="File Path:").grid(row=0, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.file_path, width=50).grid(row=0, column=1)
        tk.Button(self.root, text="Browse", command=self.browse_file).grid(row=0, column=2)
        
        tk.Label(self.root, text="Min Support Count:").grid(row=1, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.min_support_count).grid(row=1, column=1)
        
        tk.Label(self.root, text="Min Confidence (%):").grid(row=2, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.min_confidence).grid(row=2, column=1)
        
        tk.Label(self.root, text="Select Percentage of Data to Analyze (%):").grid(row=3, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.selected_percentage).grid(row=3, column=1)
        
        tk.Button(self.root, text="Run Analysis", command=self.run_analysis).grid(row=4, column=1, pady=10)
        
    def browse_file(self):
        #browse the file from the os 
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path.set(file_path)

    def filter(self,data,supp):
        #filter the data with support count 
        df = data[data.supp_count >= supp] 
        return df
        
    def count_itemset(self,trans_df, itemsets):
        count_item = {}
        for item_set in itemsets:
            #check if the item set is part of the transacrion 
            set_A = set(item_set)
            for row in trans_df:
                set_B = set(row)
                if set_B.intersection(set_A) == set_A: 
                    if item_set in count_item.keys():
                        count_item[item_set] += 1
                    else:
                        count_item[item_set] = 1
        data = pd.DataFrame()
        data['item_sets'] = count_item.keys()
        data['supp_count'] = count_item.values()
        return data

    def count_item(self,trans_items):
        count_ind_item = {}
        for row in trans_items:
            for i in range(len(row)):
                if row[i] in count_ind_item.keys():
                    count_ind_item[row[i]] += 1
                else:
                    count_ind_item[row[i]] = 1
        data = pd.DataFrame()
        data['item_sets'] = count_ind_item.keys()
        data['supp_count'] = count_ind_item.values()
        data = data.sort_values('item_sets')
        return data

    def candidates(self,list_of_items):
        itemsets = []
        i = 1
        for line in list_of_items:
            proceding = list_of_items[i:]
            #check if the item is there in the transactions
            for item in proceding:
                #first 1 item candidates 
                if(type(item) is str):
                    if line != item:
                        tuples = (line, item)
                        itemsets.append(tuples)
                else:
                    #if 3 item set check the first two item in each and if they equal append 
                    if line[:-1] == item[:-1] and line[-1] != item[-1]:
                        new_item_set = line + (item[-1],)  # Concatenate line with the last item of item
                        itemsets.append(new_item_set)
            i = i+1
        if(len(itemsets) == 0):
            return None
        return itemsets

    
    def apriori(self,trans_data,supp):
        freq = pd.DataFrame()
        df = self.count_item(trans_data)
        while(len(df) != 0):
            #check the support and elimnate 
            df = self.filter(df, supp)
            if len(df) > 1 or (len(df) == 1 and int(df.supp_count.iloc[0]) >= supp):

                freq = df
            print("item sets")
            print(df.item_sets)    
            itemsets = self.candidates(df.item_sets)
            if(itemsets is None):
                return freq
            df = self.count_itemset(trans_data, itemsets)
            
        return df
    
    def calculate_conf(self, itemset_supp_count, antecedent_supp_count):
        return round(int(itemset_supp_count )/ int(antecedent_supp_count )* 100, 2)
    def strong_rules(self, freq_item_sets, min_confidence, transactions):
        rules = []
        for idx, row in freq_item_sets.iterrows():
            items_list = row['item_sets']
            if isinstance(items_list, tuple):
                items_list = ','.join(items_list)
            items_list = items_list.split(',')
            if len(items_list) > 1:
                # make all combinations of the elements in the frequent tuple 
                subsets = list(itertools.chain.from_iterable(itertools.combinations(items_list, r) for r in range(1, len(items_list))))
                for subset in subsets:
                    if set(subset) == set(items_list):
                        continue  
                    antecedent = ','.join(subset)
                    consequent = ','.join(sorted(set(items_list) - set(subset)))
                    antecedent_supp_count = len([transaction for transaction in transactions if set(subset).issubset(set(transaction))])
                    if not antecedent or antecedent_supp_count == 0:
                        continue
                    confidence = self.calculate_conf(row['supp_count'], antecedent_supp_count)
                    if confidence >= min_confidence:
                        rules.append({'Antecedent   ': antecedent, 'Consequent   ': consequent, 'Confidence (%)': confidence})
        return pd.DataFrame(rules)


    def run_analysis(self):
        try:
            file_path = self.file_path.get()
            min_support_count = self.min_support_count.get()
            min_confidence = self.min_confidence.get() 
            selected_percentage = self.selected_percentage.get() / 100.0
            
            data = pd.read_csv(file_path)  
            num_rows = len(data)
            #part of the file (percentage)
            selected_rows = int(selected_percentage * num_rows)
            data = data.head(selected_rows)
            if data.empty:
                raise ValueError("The file does not contain any data.")

            data = data.drop_duplicates()
            transactions = data.groupby(data.columns[0])[data.columns[1]].agg(list).reset_index()
            transactions.columns = ['TID', 'item_list']
            
            freq_item_sets = self.apriori(transactions['item_list'], min_support_count)
            strong_rules_df = self.strong_rules(freq_item_sets, min_confidence, transactions["item_list"])
            #add the text 
            freq_text = freq_item_sets.to_string(index=False)
            strong_text = strong_rules_df.to_string(index=False)
            print(freq_item_sets)
            print(strong_rules_df)
            
            # Display frequent item sets and strong rules
            frm = ttk.Frame(root, padding=10)
            frm.grid()
            
            ttk.Label(frm, text="Frequent Item Sets:").grid(column=0, row=0)
            ttk.Label(frm, text=freq_text).grid(column=0, row=1)
            
            ttk.Label(frm, text="Strong Rules:").grid(column=1, row=0)
            ttk.Label(frm, text=strong_text).grid(column=1, row=1)
            #quit button 
            
            ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=2)
                        
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(str(e))


if __name__ == "__main__":
    root = tk.Tk()
    AprioriUI(root)
    #run the program 
    root.mainloop()
