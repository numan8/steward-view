from collections import Counter
from datetime import timedelta
from itertools import combinations
from typing import Iterable
from dotmap import DotMap
import pandas as pd
pd.options.mode.copy_on_write = True


class Order:
    '''
    Create a new Order instance.
    

    Parameters
    ----------
    matcher: Any Matcher instance.
    
    order_id: A datetime associated with a single order in 
    matcher.prods.original.
    '''
    
    def __init__(self, matcher, order_id):
        self.date = order_id.date()
        self.pmts = DotMap({
            'candidates': self.identify_candidates(matcher),
            'matched': pd.Index([], dtype='int64')
        })
        self.prods = DotMap({
            'unmatched': Order.extract_products(matcher, order_id),
            'matched': pd.Index([], dtype='int64')
        })
        self.counter = Counter({
            'match_all_products': 0,
            'match_single_products': 0,
            'match_product_combos': 0
        })
    
    
    def match(self):
        '''
        Identify the matching payments and products associated with a 
        single Amazon order.

        This method compares the value of candidate payments from 
        matcher.payments.filtered to the cost of 1) all products, 2)
        single products, and 3) every reasonable combination of 
        products associated with order_id. Its results are accessible
        through the following attributes:

            self.pmts.matched: An Index of matched payments
            
            self.prods.matched: An Index of matched products
            
            self.prods.unmatched: A DataFrame containing all 
            unmatched products associated with order_id
            
            self.counter: A Counter that tracks the number of product
            groups matched by the different parts of the _match method
        '''
        # Step 1
        if len(self.pmts.candidates) == 0: return None
        else: self.match_all_products()
        
        # Step 2
        if len(self.prods.unmatched) < 2: return None    # If len == 0, then the prior function matched the whole order. If len == 1, then the order contains only one product and the prior function did not find a match; the remaining functions will not find a match, either.            
        else: self.match_single_products()
        
        # Step 3
        if len(self.prods.unmatched) < 4: return None    # If len(self.products.unmatched) is 2 or 3, then at least one subset must contain only one product purchase. If this single product subset had a match, match_single_products would have found it. The function assumes in such cases that no complementary subset has a match, either.
        else: self.match_product_combos()
    
    
    def match_all_products(self):
        prods_amt = self.prods.unmatched['amount'].sum()        
        for pmt_idx, pmt_amt in self.pmts.candidates['amount'].items():
            if pmt_amt == prods_amt:
                self.record_match(
                    pd.Index([pmt_idx]), 
                    self.prods.unmatched.index, 
                    'match_all_products'
                )
                break
    
    
    def match_single_products(self):
        for pmt_idx, pmt_amt in self.pmts.candidates['amount'].items():
            for prod_idx, prod_amt in self.prods.unmatched['amount'].items():
                if prod_amt == pmt_amt:
                    self.record_match(
                        pd.Index([pmt_idx]), 
                        pd.Index([prod_idx]), 
                        'match_single_products'
                    )
                    if len(self.prods.unmatched) > 1:
                        self.match_all_products()
                    break
            if len(self.prods.unmatched) == 0:
                break
    
    
    def match_product_combos(self):
        initial_prod_count = len(self.prods.unmatched)    # This variable solves a problem in the third line of the main loop. The _record_match method removes matched products from self.products.unmatched. If used inside the loop, len(self.products.unmatched) would return a smaller value after each match. This smaller value would be the wrong operand to compare with combo_length to determine whether the loop should end. (For example, imagine an order of nine products comprising three groups of two and one group of three. This loop would first match the three groups of two and remove the associated products from self.products.unmatched. Next, the loop should increase combo_length to three, after which it would match the remaining product group in the order. However, if the loop compares len(self.products.unmatched) to combo_length, it would find that the new combo_length is equal to the number of remaining unmatched products and end the loop prematurely.)
        for combo_length in self.generate_combo_lengths(initial_prod_count):
            self.match_combos_of_length(combo_length)
            if len(self.prods.unmatched) <= combo_length:
                break
    
    
    @staticmethod
    def generate_combo_lengths(initial_prod_count):
        return range(2, initial_prod_count // 2 + 1)    # All combos must contain at least two products. If self.products.unmatched comprises multiple subsets of products with matching payments, at least one of those subsets must contain half or fewer of the unmatched products. Therefore, except for testing the combined amount of the remaining products after matching a combo, each iteration of this loop searches only for the smallest matching subset. If no subset containing half or fewer of the unmatched products matches a payment, then no larger subset will match, either (unless Amazon did something odd).        
    
    
    def match_combos_of_length(self, combo_length):
        for pmt_idx, pmt_amt in self.pmts.candidates['amount'].items():
            for combo in self.generate_combinations(combo_length):
                combo_amt = self.calculate_combo_amount(combo)
                if combo_amt == pmt_amt:
                    self.record_match(
                        pd.Index([pmt_idx]), 
                        pd.Index(combo), 
                        'match_product_combos'
                    )
                    if len(self.prods.unmatched) >= combo_length:    # >= is the correct operator for this comparison. There is no reason to generate combos of length combo_length to test against remaining payment candidates when only one such combo remains.
                        self.match_all_products()
                    break
            if len(self.prods.unmatched) <= combo_length:    # By this point, the program has tested every subset with a product count that is less than or equal to combo_length. If such a subset remains, then it has no match.
                break
    
    
    def generate_combinations(self, combo_length) -> Iterable[tuple[int]]:
        return combinations(self.prods.unmatched.index, combo_length)
    
    
    def calculate_combo_amount(self, combo):
        return self.prods.unmatched.loc[list(combo), 'amount'].sum()    
    
    
    def record_match(self, payment_index: pd.Index, product_index: pd.Index, function):
        self.pmts.matched = self.pmts.matched.append(payment_index)
        self.pmts.candidates = self.pmts.candidates.drop(index = payment_index)
        self.prods.matched = self.prods.matched.append(product_index)
        self.prods.unmatched = self.prods.unmatched.drop(index = product_index)
        self.counter[function] += 1
    
    
    def identify_candidates(self, matcher, max_delay = 3) -> pd.DataFrame:
        payments = matcher.pmts.filtered
        filter = payments['date'].between(self.date, self.date + timedelta(days = max_delay))
        return payments[filter]
    
    
    @staticmethod
    def extract_products(matcher, order_id) -> pd.DataFrame:
        products = matcher.prods.original
        products = products[products['date'] == order_id]
        return products