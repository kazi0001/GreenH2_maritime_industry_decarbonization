Set
   i / ind1*ind12 /
   j / d1,d2,GHDC, WSmme, HDem /
   counter / c1*c11 /;

Table c(i,j)

           d1              d2                GHDC        WSmme      HDem
   ind1   25.911465       51.557049          0.2         5000       4500
   ind2   25.893611       51.541222          0.33        1000       3000
   ind3   25.919639       51.491679          0.13         750       2000
   ind4   25.903227       51.52268           0.45           0       3500
   ind5   25.883797       51.556691          0.13        1200       1800
   ind6   25.91704        51.577825          0.13         500       2000
   ind7   25.9248         51.514797          0.25        3000       3000
   ind8   25.914073       51.501477          0.33           0       7500
   ind9   25.923982       51.544724          0.13           0       1300
   ind10  25.925353       51.549304          0.17        8000       2500
   ind11  25.935201       51.52617           0.13        1700       1200
   ind12  25.896211       51.518356          0.14           0       8000  ;

Variable HPcost, HScost,HcScost, HoScost, HTcost, HVcost, WDT, WSCcost, CO2Ethi,CO2Ethship, CO2etpw,CO2ET,CO2B, TCprofit,BPprofit, TAC, HPp, HSs, HD(i), HDship;
Equation eq1, eq2,eq3, eq4, eq5(j), eq6, eq7(j), eq8, eq9(j), eq10(j),eq11, eq12, eq13(j), eq14, eq15, eq16, eq17, eq18;
Parameter    report(*), rep(counter,*);
Scalar
   UHPC / 10/
   HRph  /0.1354/
   FPp /1.74/

   USFCh /9.87/


   x  /25.913/
   y  /51.533/
   LHCCt  /0.07/
   LHDChsc   /0.17/

   WPFCh  /1/
   ACFpw   /1/

   UWSCmme  /0.00148/
   UWSCw   /0.2/
   UWTCtp  /0.0027/


   CEFth  /0.023/
   CEFtpw  /0.0126/
   Fconv  /3.2/
   CO2Ef /2.75/
   CTAX   /0.32/

   UOSPhsc  /10/
   ULHChsc  /2.75/
   PDship   /36/
   ModDWT   /64772000/
*2.75
   Elim;


eq1..       HPcost =E= (HSs+HDship)*(UHPC+HRph*FPp);
eq2 ..      HScost =E= HcScost+HoScost;
eq3..       HcScost =E= (HSs+HDship)*USFCh;
eq4..       HoScost =E= HDship*USFCh;
*eq3(j)..   HTcost =E= sum((i),sqrt(sqr(c(i,'d1')-(x))+sqr(c(i,'d2')-(y)))*c(i,'GHDC')*HD(i)) + (LHCCt+LHDChsc)*LHShsc;
eq5(j)..    HTcost =E= sum((i),sqrt(sqr(c(i,'d1')/25-(x)/25)+ sqr(c(i,'d2')/51-(y))/51)*c(i,'GHDC')*HD(i)) + (LHCCt+LHDChsc)*HDship;
eq6..       WDT    =E= 0.375*HPp;
eq7(j)..    WSCcost =E= UWSCmme*WDT+ UWSCmme*sum(i,c(i,'WSmme'))+UWTCtp$(sum(i,c(i,'WSmme'))<15000)*(WDT-sum(i,c(i,'WSmme')))+0.2*UWSCw*WDT;
*eq6(j)..   WSCcost$(WDT>sum(i,c(i,'WSmme'))) =E= UWSCmme*sum(i,c(i,'WSmme'))+UWTCtp*(WDT-sum(i,c(i,'WSmme')))+0.2*UWSCw*WDT;
eq8..       sum(i,HD(i))=E= HSs;
eq9(j)..    CO2Ethi =E= sum(i, HD(i))*CEFth;
eq10(j)..   CO2etpw =E= (WDT-sum(i,c(i,'WSmme'))$(sum(i,c(i,'WSmme'))<15000))*CEFtpw;
eq11 ..     CO2Ethship =E= HDship*CEFth;
eq12..      CO2ET =E= CO2Ethi+CO2Ethship+CO2etpw;
eq13(j)..   CO2B =E= sum(i,HD(i))*Fconv*CO2Ef + HDship*Fconv*CO2Ef;
eq14..      TCprofit =E= (CO2B-CO2ET)*CTAX;
eq15..      BPprofit =E= 0.33*HPp*UOSPhsc + HDship*ULHChsc;
eq16..      HPp =E= sum(i,HD(i))+HDship;
eq17..      HVcost =E= PDship*ModDWT*5;

eq18..      TAC =E= HPcost + HScost + HTcost + HVcost + WSCcost - TCprofit - BPprofit;


*5% = HPp.lo 353028000;
*5% = HDship.lo 3280000000;

*HPp.lo = 3280000000;
*HPp.up = 2*3280000000;
HSs.lo = 353028000;
HSs.up = 2*353028000;
HD.lo(i) = 0;
HD.up(i) = c(i,'HDem')*24*365;
*HD.lo(i) = 353028000;
*HD.up(i) = 1*353028000;
HDship.lo = 5*3280000000;
HDship.up = 6*3280000000;

*chainge eq17, HDshioplo and high

Model minlp1 / all /;
solve minlp1 using MINLP minimizing TAC;

Display HPcost.l, HScost.l, HcScost.l, HoScost.l, HTcost.l,HVcost.l, WSCcost.l,TCprofit.l, BPprofit.l, TAC.l, WDT.l, HPp.l, HSs.l, HDship.l,CO2Ethi.l, CO2etpw.l, CO2ET.l, CO2B.l, HD.l;

Model OBJ1 / all /;
