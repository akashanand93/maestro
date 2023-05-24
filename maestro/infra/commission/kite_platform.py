from backtrader import CommInfoBase

class ZerodhaCommission(CommInfoBase):
    params = (
        ('stamp_duty', 0.003),  # 0.003% or ₹300 / crore on buy side
        ('stt', 0.025),  # 0.025% on the sell side
        ('transaction_charges', 0.00325),  # NSE: 0.00325%, BSE: 0.00375%
        ('gst_rate', 0.18),  # GST 18% on (brokerage + SEBI charges + transaction charges)
        ('sebi_charges', 0.0000001),  # ₹10 / crore
        ('brokerage', 0.03),  # 0.03% or Rs. 20/executed order whichever is lower
        ('flat_rate', 20),  # Flat rate of Rs. 20
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        Calculate the commission for this order execution.
        """
        # Brokerage calculation
        brokerage_cost = min(abs(size) * price * self.p.brokerage / 100, self.p.flat_rate)
        # SEBI Charges
        sebi_cost = abs(size) * price * self.p.sebi_charges
        # Transaction charges
        transaction_cost = abs(size) * price * self.p.transaction_charges / 100
        # GST calculation
        gst = self.p.gst_rate * (brokerage_cost + sebi_cost + transaction_cost)
        # STT/CTT calculation
        stt_cost = abs(size) * price * self.p.stt / 100 if size < 0 else 0  # Only on sell side
        # Stamp Duty calculation
        stamp_cost = abs(size) * price * self.p.stamp_duty / 100 if size > 0 else 0  # Only on buy side

        total = brokerage_cost + sebi_cost + transaction_cost + gst + stt_cost + stamp_cost

        #print("Brokerage: ", brokerage_cost, 
        #      "SEBI: ", sebi_cost, 
        #      "Transaction: ", transaction_cost, 
        #      "GST: ", gst, 
        #      "STT: ", stt_cost,
        #      "Stamp: ", stamp_cost,
        #      "total: ", total)

        return total
