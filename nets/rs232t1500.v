`timescale 1ns/100ps

module uart ( sys_clk, sys_rst_l, uart_XMIT_dataH, xmitH, xmit_dataH, 
        xmit_doneH, uart_REC_dataH, rec_dataH, rec_readyH, test_mode, test_se, 
        test_si, test_so );
  input [7:0] xmit_dataH;
  output [7:0] rec_dataH;
  input sys_clk, sys_rst_l, xmitH, uart_REC_dataH, test_mode, test_se, test_si;
  output uart_XMIT_dataH, xmit_doneH, rec_readyH, test_so;
  wire   iXMIT_xmit_doneInH, iXMIT_next_state_0_, iXMIT_next_state_1_,
         iXMIT_next_state_2_, iXMIT_state_0_, iXMIT_state_1_, iXMIT_state_2_,
         iXMIT_N46, iXMIT_N45, iXMIT_N44, iXMIT_bitCountH_0_,
         iXMIT_bitCountH_1_, iXMIT_bitCountH_2_, iXMIT_bitCountH_3_, iXMIT_N29,
         iXMIT_N28, iXMIT_N27, iXMIT_N26, iXMIT_N25, iXMIT_N24, iXMIT_N23,
         iXMIT_bitCell_cntrH_0_, iXMIT_bitCell_cntrH_1_,
         iXMIT_bitCell_cntrH_2_, iXMIT_bitCell_cntrH_3_,
         iXMIT_xmit_ShiftRegH_1_, iXMIT_xmit_ShiftRegH_2_,
         iXMIT_xmit_ShiftRegH_3_, iXMIT_xmit_ShiftRegH_4_,
         iXMIT_xmit_ShiftRegH_5_, iXMIT_xmit_ShiftRegH_6_,
         iXMIT_xmit_ShiftRegH_7_, iRECEIVER_rec_readyInH,
         iRECEIVER_next_state_0_, iRECEIVER_next_state_1_,
         iRECEIVER_next_state_2_, iRECEIVER_state_0_, iRECEIVER_state_1_,
         iRECEIVER_state_2_, iRECEIVER_N28, iRECEIVER_N27, iRECEIVER_N26,
         iRECEIVER_recd_bitCntrH_0_, iRECEIVER_recd_bitCntrH_1_,
         iRECEIVER_recd_bitCntrH_2_, iRECEIVER_recd_bitCntrH_3_, iRECEIVER_N23,
         iRECEIVER_N22, iRECEIVER_N21, iRECEIVER_N20, iRECEIVER_N19,
         iRECEIVER_N18, iRECEIVER_N17, iRECEIVER_bitCell_cntrH_0_,
         iRECEIVER_bitCell_cntrH_1_, iRECEIVER_bitCell_cntrH_2_,
         iRECEIVER_bitCell_cntrH_3_, iRECEIVER_rec_datSyncH,
         iRECEIVER_rec_datH, n2, n3, n7, n8, n9, n10, n11, n12, n15, n17, n18,
         n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n31, n33, n35,
         n37, n39, n41, n42, n43, n44, n45, n46, n47, n48, n49, n50, n51, n52,
         n53, n54, n55, n56, n57, n58, n60, n61, n62, n63, n64, n65, n66, n67,
         n68, n69, n71, n72, n73, n74, n75, n77, n79, n80, n81, n82, n83, n84,
         n85, n86, n87, n88, n89, n91, n92, n93, n94, n100, n103, n106, n109,
         n116, n123, n130, n137, n144, n151, n158, n165, n190, n193, n196,
         n199, n202, n205, n208, n211, n214, n217, n220, n223, n238, n239,
         n241, n242, n243, n244, n245, n246, n247, n248, n249, n250, n251,
         n252, n253, n254, n255, n256, n257, n258, n259, n260, n261, n262,
         n263, n264, n265, n266, n267, n268, n269, n270, n271, n272, n273,
         n274, n275, \test_point/DOUT , \test_point/TM , n281, n370, n371;

  wire iXMIT_state_CTRL, iXMIT_N_CTRL_1_, iXMIT_N_CTRL_2_, iXMIT_xmit_CTRL, iXMIT_CRTL,
       iRECEIVER_state_CTRL,iRECEIVER_N_CTRL_1_,iRECEIVER_N_CTRL_2_,iRECEIVER_bitCell_CTRL, iRECEIVER_CTRL,
       iCTRL,xmit_doneH_temp,uart_XMIT_dataH_temp;

  wire   [7:0] rec_dataH_rec;
  wire   [7:0] rec_dataH_temp;
  assign test_so = rec_dataH_temp[7];
  assign \test_point/TM  = test_mode;

OAI21X1_RS U3 ( .A0(iXMIT_state_2_), .A1(iXMIT_state_1_), .B0(n2), .Y(
        uart_XMIT_dataH) );
AOI21X1_RS U4 ( .A0(iXMIT_state_2_), .A1(iXMIT_state_0_), .B0(n3), .Y(n2) );
AOI21X1_RS U5 ( .A0(n242), .A1(n239), .B0(n257), .Y(n3) );
INVX1_RS U6 ( .A(n7), .Y(n100) );
AOI22X1_RS U7 ( .A0(iRECEIVER_N27), .A1(n8), .B0(iRECEIVER_recd_bitCntrH_2_),
        .B1(n9), .Y(n7) );
INVX1_RS U8 ( .A(n10), .Y(n103) );
AOI22X1_RS U9 ( .A0(iRECEIVER_N26), .A1(n8), .B0(iRECEIVER_recd_bitCntrH_1_),
        .B1(n9), .Y(n10) );
INVX1_RS U10 ( .A(n11), .Y(n106) );
AOI22X1_RS U11 ( .A0(n243), .A1(n8), .B0(iRECEIVER_recd_bitCntrH_0_), .B1(n9),
        .Y(n11) );
INVX1_RS U12 ( .A(n12), .Y(n109) );
AOI22X1_RS U13 ( .A0(iRECEIVER_N28), .A1(n8), .B0(iRECEIVER_recd_bitCntrH_3_),
        .B1(n9), .Y(n12) );
OAI21X1_RS U14 ( .A0(n238), .A1(n245), .B0(n15), .Y(n9) );
OAI21X1_RS U15 ( .A0(iRECEIVER_state_1_), .A1(n248), .B0(n238), .Y(n15) );
INVX1_RS U16 ( .A(n17), .Y(n116) );
AOI22X1_RS U17 ( .A0(rec_dataH_rec[1]), .A1(n8), .B0(rec_dataH_rec_0_temp), .B1(n18), .Y(n17) );
INVX1_RS U18 ( .A(n19), .Y(n123) );
AOI22X1_RS U19 ( .A0(rec_dataH_rec[2]), .A1(n8), .B0(rec_dataH_rec[1]), .B1(n18), .Y(n19) );
INVX1_RS U20 ( .A(n20), .Y(n130) );
AOI22X1_RS U21 ( .A0(rec_dataH_rec[3]), .A1(n8), .B0(rec_dataH_rec[2]), .B1(n18), .Y(n20) );
INVX1_RS U22 ( .A(n21), .Y(n137) );
AOI22X1_RS U23 ( .A0(rec_dataH_rec[4]), .A1(n8), .B0(rec_dataH_rec[3]), .B1(n18), .Y(n21) );
INVX1_RS U24 ( .A(n22), .Y(n144) );
AOI22X1_RS U25 ( .A0(rec_dataH_rec[5]), .A1(n8), .B0(rec_dataH_rec[4]), .B1(n18), .Y(n22) );
INVX1_RS U26 ( .A(n23), .Y(n151) );
AOI22X1_RS U27 ( .A0(rec_dataH_rec[6]), .A1(n8), .B0(rec_dataH_rec[5]), .B1(n18), .Y(n23) );
INVX1_RS U28 ( .A(n24), .Y(n158) );
AOI22X1_RS U29 ( .A0(rec_dataH_rec[7]), .A1(n8), .B0(rec_dataH_rec[6]), .B1(n18), .Y(n24) );
INVX1_RS U30 ( .A(n25), .Y(n165) );
AOI22X1_RS U31 ( .A0(iRECEIVER_rec_datH), .A1(n8), .B0(rec_dataH_rec[7]), .B1(
        n18), .Y(n25) );
OAI21X1_RS U32 ( .A0(n257), .A1(n26), .B0(n27), .Y(n190) );
AOI22X1_RS U33 ( .A0(xmit_dataH[0]), .A1(n28), .B0(iXMIT_xmit_ShiftRegH_1_),
        .B1(n29), .Y(n27) );
OAI21X1_RS U35 ( .A0(n263), .A1(n26), .B0(n31), .Y(n193) );
AOI22X1_RS U36 ( .A0(xmit_dataH[1]), .A1(n28), .B0(iXMIT_xmit_ShiftRegH_2_),
        .B1(n29), .Y(n31) );
OAI21X1_RS U38 ( .A0(n26), .A1(n262), .B0(n33), .Y(n196) );
AOI22X1_RS U39 ( .A0(xmit_dataH[2]), .A1(n28), .B0(iXMIT_xmit_ShiftRegH_3_),
        .B1(n29), .Y(n33) );
OAI21X1_RS U41 ( .A0(n26), .A1(n261), .B0(n35), .Y(n199) );
AOI22X1_RS U42 ( .A0(xmit_dataH[3]), .A1(n28), .B0(iXMIT_xmit_ShiftRegH_4_),
        .B1(n29), .Y(n35) );
OAI21X1_RS U44 ( .A0(n26), .A1(n260), .B0(n37), .Y(n202) );
AOI22X1_RS U45 ( .A0(xmit_dataH[4]), .A1(n28), .B0(iXMIT_xmit_ShiftRegH_5_),
        .B1(n29), .Y(n37) );
OAI21X1_RS U47 ( .A0(n26), .A1(n259), .B0(n39), .Y(n205) );
AOI22X1_RS U48 ( .A0(xmit_dataH[5]), .A1(n28), .B0(iXMIT_xmit_ShiftRegH_6_),
        .B1(n29), .Y(n39) );
OAI21X1_RS U50 ( .A0(n26), .A1(n258), .B0(n41), .Y(n208) );
AOI22X1_RS U51 ( .A0(xmit_dataH[6]), .A1(n28), .B0(iXMIT_xmit_ShiftRegH_7_),
        .B1(n29), .Y(n41) );
INVX1_RS U55 ( .A(n44), .Y(n26) );
AOI21X1_RS U56 ( .A0(iXMIT_xmit_ShiftRegH_7_), .A1(n44), .B0(n29), .Y(n42) );
NOR2X1_RS U57 ( .A(n44), .B(n28), .Y(n29) );
INVX1_RS U58 ( .A(n45), .Y(n214) );
AOI22X1_RS U59 ( .A0(n46), .A1(iXMIT_bitCountH_3_), .B0(iXMIT_N46), .B1(n47),
        .Y(n45) );
INVX1_RS U60 ( .A(n48), .Y(n217) );
AOI22X1_RS U61 ( .A0(iXMIT_bitCountH_2_), .A1(n46), .B0(iXMIT_N45), .B1(n47),
        .Y(n48) );
INVX1_RS U62 ( .A(n49), .Y(n220) );
AOI22X1_RS U63 ( .A0(iXMIT_bitCountH_1_), .A1(n46), .B0(iXMIT_N44), .B1(n47),
        .Y(n49) );
INVX1_RS U64 ( .A(n50), .Y(n223) );
AOI22X1_RS U65 ( .A0(iXMIT_bitCountH_0_), .A1(n46), .B0(n252), .B1(n47), .Y(n50) );
NOR2X1_RS U66 ( .A(n51), .B(n47), .Y(n46) );
NOR2X1_RS U67 ( .A(n52), .B(n53), .Y(n47) );
INVX1_RS U68 ( .A(n54), .Y(n52) );
INVX1_RS U69 ( .A(n55), .Y(n51) );
NAND3X1_RS U71 ( .A(iXMIT_state_2_), .B(iXMIT_state_0_), .C(n57), .Y(n56) );
NAND3X1_RS U72 ( .A(n242), .B(n58), .C(n246), .Y(n55) );
INVX1_RS U73 ( .A(xmitH), .Y(n58) );
INVX1_RS U74 ( .A(n60), .Y(iXMIT_next_state_2_) );
AOI22X1_RS U75 ( .A0(n54), .A1(n61), .B0(n62), .B1(iXMIT_state_0_), .Y(n60) );
AOI21X1_RS U77 ( .A0(n239), .A1(iXMIT_state_2_), .B0(n28), .Y(n44) );
NAND3X1_RS U79 ( .A(n246), .B(n242), .C(xmitH), .Y(n64) );
AOI21X1_RS U80 ( .A0(iXMIT_state_1_temp), .A1(n65), .B0(n66), .Y(n63) );
AOI21X1_RS U82 ( .A0(n66), .A1(n57), .B0(n62), .Y(n68) );
AOI22X1_RS U83 ( .A0(n54), .A1(n53), .B0(iXMIT_state_2_), .B1(n239), .Y(n67) );
OR4X1_RS U85 ( .A(n253), .B(iXMIT_bitCountH_0_), .C(iXMIT_bitCountH_1_), .D(
        iXMIT_bitCountH_2_), .Y(n69) );
AND2X1_RS U87 ( .A(iXMIT_N25), .B(n71), .Y(iXMIT_N29) );
AND2X1_RS U88 ( .A(iXMIT_N24), .B(n71), .Y(iXMIT_N28) );
AND2X1_RS U89 ( .A(iXMIT_N23), .B(n71), .Y(iXMIT_N27) );
AND2X1_RS U90 ( .A(n254), .B(n71), .Y(iXMIT_N26) );
OAI21X1_RS U91 ( .A0(n239), .A1(n72), .B0(n73), .Y(n71) );
AOI22X1_RS U92 ( .A0(n66), .A1(n74), .B0(n54), .B1(n65), .Y(n73) );
INVX1_RS U93 ( .A(n61), .Y(n65) );
NOR2X1_RS U94 ( .A(n75), .B(iXMIT_bitCell_cntrH_0_), .Y(n61) );
NOR2X1_RS U95 ( .A(n246), .B(n239), .Y(n54) );
INVX1_RS U96 ( .A(n57), .Y(n74) );
NOR2X1_RS U97 ( .A(n246), .B(iXMIT_state_0_), .Y(n66) );
INVX1_RS U99 ( .A(n62), .Y(n72) );
NOR2X1_RS U100 ( .A(n242), .B(n57), .Y(n62) );
NOR2X1_RS U101 ( .A(n254), .B(n75), .Y(n57) );
NAND3X1_RS U102 ( .A(iXMIT_bitCell_cntrH_2_), .B(iXMIT_bitCell_cntrH_1_), .C(
        iXMIT_bitCell_cntrH_3_), .Y(n75) );
OAI21X1_RS U106 ( .A0(n238), .A1(n245), .B0(n77), .Y(iRECEIVER_rec_readyInH) );
NAND3X1_RS U107 ( .A(n241), .B(n238), .C(iRECEIVER_rec_datH), .Y(n77) );
AND2X1_RS U108 ( .A(n79), .B(n80), .Y(iRECEIVER_next_state_2_) );
NAND4X1_RS U109 ( .A(n81), .B(n82), .C(n18), .D(n83), .Y(
        iRECEIVER_next_state_1_) );
NOR2X1_RS U111 ( .A(n238), .B(iRECEIVER_state_0_), .Y(n8) );
NAND3X1_RS U112 ( .A(n241), .B(n238), .C(n248), .Y(n82) );
OR2X1_RS U113 ( .A(iRECEIVER_state_0_), .B(n84), .Y(n81) );
AOI21X1_RS U114 ( .A0(iRECEIVER_state_1_), .A1(n85), .B0(n248), .Y(n84) );
NAND4X1_RS U116 ( .A(n83), .B(n238), .C(n86), .D(n87), .Y(
        iRECEIVER_next_state_0_) );
AOI22X1_RS U117 ( .A0(n88), .A1(iRECEIVER_state_1_), .B0(iRECEIVER_rec_datH),
        .B1(n241), .Y(n87) );
NAND4X1_RS U118 ( .A(n80), .B(iRECEIVER_recd_bitCntrH_3_), .C(n89), .D(n243),
        .Y(n86) );
NOR2X1_RS U120 ( .A(iRECEIVER_recd_bitCntrH_2_), .B(iRECEIVER_recd_bitCntrH_1_),
        .Y(n89) );
NOR2X1_RS U123 ( .A(n241), .B(n245), .Y(n80) );
AND2X1_RS U124 ( .A(iRECEIVER_N19), .B(n92), .Y(iRECEIVER_N23) );
AND2X1_RS U125 ( .A(iRECEIVER_N18), .B(n92), .Y(iRECEIVER_N22) );
AND2X1_RS U126 ( .A(iRECEIVER_N17), .B(n92), .Y(iRECEIVER_N21) );
AND2X1_RS U127 ( .A(n255), .B(n92), .Y(iRECEIVER_N20) );
AND2X1_RS U128 ( .A(n93), .B(n94), .Y(n92) );
AOI21X1_RS U129 ( .A0(n88), .A1(n245), .B0(iRECEIVER_state_2_), .Y(n94) );
INVX1_RS U131 ( .A(n85), .Y(n88) );
NAND4X1_RS U132 ( .A(iRECEIVER_bitCell_cntrH_2_), .B(n255), .C(n244), .D(n256),
        .Y(n85) );
AOI21X1_RS U135 ( .A0(n79), .A1(iRECEIVER_state_0_), .B0(n241), .Y(n93) );
INVX1_RS U137 ( .A(n91), .Y(n79) );
NAND4X1_RS U138 ( .A(iRECEIVER_bitCell_cntrH_3_), .B(iRECEIVER_bitCell_cntrH_2_), .C(iRECEIVER_bitCell_cntrH_1_), .D(n255), .Y(n91) );
INVX1_RS U198 ( .A(n267), .Y(n265) );
INVX1_RS U199 ( .A(n267), .Y(n266) );
INVX1_RS U200 ( .A(n267), .Y(n264) );
NAND2X1_RS U201 ( .A(n80), .B(n91), .Y(n83) );
INVX1_RS U202 ( .A(n8), .Y(n18) );
INVX1_RS U203 ( .A(sys_rst_l), .Y(n267) );
NAND2X1_RS U204 ( .A(n67), .B(n68), .Y(iXMIT_next_state_0_) );
NAND2X1_RS U205 ( .A(n42), .B(n43), .Y(n211) );
NAND2X1_RS U206 ( .A(xmit_dataH[7]), .B(n26), .Y(n43) );
NAND2X1_RS U207 ( .A(n63), .B(n44), .Y(iXMIT_next_state_1_) );
NAND2X1_RS U208 ( .A(iXMIT_bitCountH_1_), .B(iXMIT_bitCountH_0_), .Y(n270) );
NAND2X1_RS U209 ( .A(n61), .B(n69), .Y(n53) );
NAND2X1_RS U210 ( .A(iRECEIVER_recd_bitCntrH_1_), .B(iRECEIVER_recd_bitCntrH_0_), .Y(n274) );
NAND2X1_RS U211 ( .A(iXMIT_bitCell_cntrH_1_), .B(iXMIT_bitCell_cntrH_0_), .Y(
        n268) );
NAND2X1_RS U212 ( .A(iRECEIVER_bitCell_cntrH_1_), .B(iRECEIVER_bitCell_cntrH_0_), .Y(n272) );
NAND2X1_RS U213 ( .A(n55), .B(n56), .Y(iXMIT_xmit_doneInH) );
INVX1_RS U214 ( .A(n64), .Y(n28) );
XOR2X1_RS U215 ( .A(iXMIT_bitCell_cntrH_1_), .B(iXMIT_bitCell_cntrH_0_), .Y(
        iXMIT_N23) );
XOR2X1_RS U216 ( .A(n251), .B(n268), .Y(iXMIT_N24) );
NOR2X1_RS U217 ( .A(n268), .B(n251), .Y(n269) );
XOR2X1_RS U218 ( .A(iXMIT_bitCell_cntrH_3_), .B(n269), .Y(iXMIT_N25) );
XOR2X1_RS U219 ( .A(iXMIT_bitCountH_1_), .B(iXMIT_bitCountH_0_), .Y(iXMIT_N44)
         );
XOR2X1_RS U220 ( .A(n247), .B(n270), .Y(iXMIT_N45) );
NOR2X1_RS U221 ( .A(n270), .B(n247), .Y(n271) );
XOR2X1_RS U222 ( .A(iXMIT_bitCountH_3_), .B(n271), .Y(iXMIT_N46) );
XOR2X1_RS U223 ( .A(iRECEIVER_bitCell_cntrH_1_), .B(iRECEIVER_bitCell_cntrH_0_),
        .Y(iRECEIVER_N17) );
XOR2X1_RS U224 ( .A(n249), .B(n272), .Y(iRECEIVER_N18) );
NOR2X1_RS U225 ( .A(n272), .B(n249), .Y(n273) );
XOR2X1_RS U226 ( .A(iRECEIVER_bitCell_cntrH_3_), .B(n273), .Y(iRECEIVER_N19) );
XOR2X1_RS U227 ( .A(iRECEIVER_recd_bitCntrH_1_), .B(iRECEIVER_recd_bitCntrH_0_),
        .Y(iRECEIVER_N26) );
XOR2X1_RS U228 ( .A(n250), .B(n274), .Y(iRECEIVER_N27) );
NOR2X1_RS U229 ( .A(n274), .B(n250), .Y(n275) );
XOR2X1_RS U230 ( .A(iRECEIVER_recd_bitCntrH_3_), .B(n275), .Y(iRECEIVER_N28) );
SDFFSRX1_RS iXMIT_bitCell_cntrH_reg_2_ ( .D(iXMIT_N28), .SI(
        iXMIT_bitCell_cntrH_1_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(
        n266), .Q(iXMIT_bitCell_cntrH_2_), .QN(n251) );
SDFFSRX1_RS iXMIT_state_reg_0_ ( .D(iXMIT_next_state_0_), .SI(
        iXMIT_bitCountH_3_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(
        sys_rst_l), .Q(iXMIT_state_0_), .QN(n239) );
SDFFSRX1_RS iXMIT_state_reg_2_ ( .D(iXMIT_next_state_2_), .SI(iXMIT_state_1_temp),
        .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(sys_rst_l), .Q(
        iXMIT_state_2_), .QN(n242) );
SDFFSRX1_RS iXMIT_state_reg_1_ ( .D(iXMIT_next_state_1_), .SI(iXMIT_state_0_),
        .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(sys_rst_l), .Q(
        iXMIT_state_1_temp), .QN(n246) );
SDFFSRX1_RS iXMIT_bitCountH_reg_0_ ( .D(n223), .SI(iXMIT_bitCell_cntrH_3_),
        .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(n266), .Q(
        iXMIT_bitCountH_0_), .QN(n252) );
SDFFSRX1_RS iXMIT_bitCountH_reg_1_ ( .D(n220), .SI(iXMIT_bitCountH_0_), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n265), .Q(iXMIT_bitCountH_1_)
         );
SDFFSRX1_RS iXMIT_bitCountH_reg_2_ ( .D(n217), .SI(iXMIT_bitCountH_1_), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n264), .Q(iXMIT_bitCountH_2_), 
        .QN(n247) );
SDFFSRX1_RS iXMIT_bitCountH_reg_3_ ( .D(n214), .SI(iXMIT_bitCountH_2_), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n266), .Q(iXMIT_bitCountH_3_), 
        .QN(n253) );
SDFFSRX1_RS iXMIT_xmit_ShiftRegH_reg_7_ ( .D(n211), .SI(iXMIT_xmit_ShiftRegH_6_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(n266), .Q(iXMIT_xmit_ShiftRegH_7_) );
SDFFSRX1_RS iXMIT_xmit_ShiftRegH_reg_6_ ( .D(n208), .SI(iXMIT_xmit_ShiftRegH_5_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(n266), .Q(iXMIT_xmit_ShiftRegH_6_), .QN(n258) );
SDFFSRX1_RS iXMIT_xmit_ShiftRegH_reg_5_ ( .D(n205), .SI(iXMIT_xmit_ShiftRegH_4_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(n266), .Q(iXMIT_xmit_ShiftRegH_5_), .QN(n259) );
SDFFSRX1_RS iXMIT_xmit_ShiftRegH_reg_4_ ( .D(n202), .SI(iXMIT_xmit_ShiftRegH_3_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(n266), .Q(iXMIT_xmit_ShiftRegH_4_), .QN(n260) );
SDFFSRX1_RS iXMIT_xmit_ShiftRegH_reg_3_ ( .D(n199), .SI(iXMIT_xmit_ShiftRegH_2_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(n266), .Q(iXMIT_xmit_ShiftRegH_3_), .QN(n261) );
SDFFSRX1_RS iXMIT_xmit_ShiftRegH_reg_2_ ( .D(n196), .SI(iXMIT_xmit_ShiftRegH_1_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(n266), .Q(iXMIT_xmit_ShiftRegH_2_), .QN(n262) );
SDFFSRX1_RS iXMIT_xmit_ShiftRegH_reg_1_ ( .D(n193), .SI(n281), .SE(test_se),
        .CK(sys_clk), .SN(1'b1), .RN(n266), .Q(iXMIT_xmit_ShiftRegH_1_), .QN(
        n263) );
SDFFSRX1_RS iXMIT_xmit_doneH_reg ( .D(iXMIT_xmit_doneInH), .SI(
        iXMIT_xmit_ShiftRegH_7_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(
        n266), .Q(xmit_doneH_temp) );
SDFFSRX1_RS iRECEIVER_state_reg_1_ ( .D(iRECEIVER_next_state_1_), .SI(
        iRECEIVER_state_0_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(n265), 
        .Q(iRECEIVER_state_1_), .QN(n241) );
SDFFSRX1_RS iRECEIVER_state_reg_0_ ( .D(iRECEIVER_next_state_0_), .SI(
        iRECEIVER_recd_bitCntrH_3_), .SE(test_se), .CK(sys_clk), .SN(n265), 
        .RN(1'b1), .Q(iRECEIVER_state_0_), .QN(n245) );
SDFFSRX1_RS iRECEIVER_bitCell_cntrH_reg_0_ ( .D(iRECEIVER_N20), .SI(test_si),
        .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(n265), .Q(
        iRECEIVER_bitCell_cntrH_0_), .QN(n255) );
SDFFSRX1_RS iRECEIVER_bitCell_cntrH_reg_1_ ( .D(iRECEIVER_N21), .SI(
        iRECEIVER_bitCell_cntrH_0_), .SE(test_se), .CK(sys_clk), .SN(1'b1), 
        .RN(n265), .Q(iRECEIVER_bitCell_cntrH_1_), .QN(n244) );
SDFFSRX1_RS iRECEIVER_bitCell_cntrH_reg_2_ ( .D(iRECEIVER_N22), .SI(
        iRECEIVER_bitCell_cntrH_1_), .SE(test_se), .CK(sys_clk), .SN(1'b1), 
        .RN(n265), .Q(iRECEIVER_bitCell_cntrH_2_), .QN(n249) );
SDFFSRX1_RS iRECEIVER_bitCell_cntrH_reg_3_ ( .D(iRECEIVER_N23), .SI(
        iRECEIVER_bitCell_cntrH_2_), .SE(test_se), .CK(sys_clk), .SN(1'b1), 
        .RN(n265), .Q(iRECEIVER_bitCell_cntrH_3_), .QN(n256) );
SDFFSRX1_RS iRECEIVER_state_reg_2_ ( .D(iRECEIVER_next_state_2_), .SI(
        iRECEIVER_state_1_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(n265), 
        .Q(iRECEIVER_state_2_), .QN(n238) );
SDFFSRX1_RS iRECEIVER_rec_readyH_reg ( .D(iRECEIVER_rec_readyInH), .SI(
        iRECEIVER_rec_datSyncH), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(
        n265), .Q(rec_readyH) );
SDFFSRX1_RS iRECEIVER_par_dataH_reg_7_ ( .D(n165), .SI(rec_dataH_rec[6]), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n265), .Q(rec_dataH_rec[7]) );
SDFFSRX1_RS rec_dataH_temp_reg_7_ ( .D(rec_dataH_rec[7]), .SI(rec_dataH_temp[6]), .SE(test_se), .CK(\test_point/DOUT ), .SN(1'b1), .RN(n265), .Q(
        rec_dataH_temp[7]) );
SDFFSRX1_RS rec_dataH_reg_7_ ( .D(rec_dataH_temp[7]), .SI(rec_dataH[6]), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n265), .Q(rec_dataH[7]) );
SDFFSRX1_RS iRECEIVER_par_dataH_reg_6_ ( .D(n158), .SI(rec_dataH_rec[5]), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n265), .Q(rec_dataH_rec[6]) );
SDFFSRX1_RS rec_dataH_temp_reg_6_ ( .D(rec_dataH_rec[6]), .SI(rec_dataH_temp[5]), .SE(test_se), .CK(\test_point/DOUT ), .SN(1'b1), .RN(n265), .Q(
        rec_dataH_temp[6]) );
SDFFSRX1_RS rec_dataH_reg_6_ ( .D(rec_dataH_temp[6]), .SI(rec_dataH[5]), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n264), .Q(rec_dataH[6]) );
SDFFSRX1_RS iRECEIVER_par_dataH_reg_5_ ( .D(n151), .SI(rec_dataH_rec[4]), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n266), .Q(rec_dataH_rec[5]) );
SDFFSRX1_RS rec_dataH_temp_reg_5_ ( .D(rec_dataH_rec[5]), .SI(rec_dataH_temp[4]), .SE(test_se), .CK(\test_point/DOUT ), .SN(1'b1), .RN(n266), .Q(
        rec_dataH_temp[5]) );
SDFFSRX1_RS rec_dataH_reg_5_ ( .D(rec_dataH_temp[5]), .SI(rec_dataH[4]), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n265), .Q(rec_dataH[5]) );
SDFFSRX1_RS iRECEIVER_par_dataH_reg_4_ ( .D(n144), .SI(rec_dataH_rec[3]), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n266), .Q(rec_dataH_rec[4]) );
SDFFSRX1_RS rec_dataH_temp_reg_4_ ( .D(rec_dataH_rec[4]), .SI(rec_dataH_temp[3]), .SE(test_se), .CK(\test_point/DOUT ), .SN(1'b1), .RN(n264), .Q(
        rec_dataH_temp[4]) );
SDFFSRX1_RS rec_dataH_reg_4_ ( .D(rec_dataH_temp[4]), .SI(rec_dataH[3]), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n266), .Q(rec_dataH[4]) );
SDFFSRX1_RS iRECEIVER_par_dataH_reg_3_ ( .D(n137), .SI(rec_dataH_rec[2]), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n265), .Q(rec_dataH_rec[3]) );
SDFFSRX1_RS rec_dataH_temp_reg_3_ ( .D(rec_dataH_rec[3]), .SI(rec_dataH_temp[2]), .SE(test_se), .CK(\test_point/DOUT ), .SN(1'b1), .RN(n264), .Q(
        rec_dataH_temp[3]) );
SDFFSRX1_RS rec_dataH_reg_3_ ( .D(rec_dataH_temp[3]), .SI(rec_dataH[2]), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n265), .Q(rec_dataH[3]) );
SDFFSRX1_RS iRECEIVER_par_dataH_reg_2_ ( .D(n130), .SI(rec_dataH_rec[1]), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n264), .Q(rec_dataH_rec[2]) );
SDFFSRX1_RS rec_dataH_temp_reg_2_ ( .D(rec_dataH_rec[2]), .SI(rec_dataH_temp[1]), .SE(test_se), .CK(\test_point/DOUT ), .SN(1'b1), .RN(n264), .Q(
        rec_dataH_temp[2]) );
SDFFSRX1_RS rec_dataH_reg_2_ ( .D(rec_dataH_temp[2]), .SI(rec_dataH[1]), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n264), .Q(rec_dataH[2]) );
SDFFSRX1_RS iRECEIVER_par_dataH_reg_1_ ( .D(n123), .SI(rec_dataH_rec_0_temp), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n264), .Q(rec_dataH_rec[1]) );
SDFFSRX1_RS rec_dataH_temp_reg_1_ ( .D(rec_dataH_rec[1]), .SI(rec_dataH_temp[0]), .SE(test_se), .CK(\test_point/DOUT ), .SN(1'b1), .RN(n264), .Q(
        rec_dataH_temp[1]) );
SDFFSRX1_RS rec_dataH_reg_1_ ( .D(rec_dataH_temp[1]), .SI(rec_dataH[0]), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n264), .Q(rec_dataH[1]) );
SDFFSRX1_RS iRECEIVER_par_dataH_reg_0_ ( .D(n116), .SI(
        iRECEIVER_bitCell_cntrH_3_), .SE(test_se), .CK(sys_clk), .SN(1'b1), 
        .RN(n264), .Q(rec_dataH_rec_0_temp) );
SDFFSRX1_RS rec_dataH_temp_reg_0_ ( .D(rec_dataH_rec_0_temp), .SI(rec_dataH[7]),
        .SE(test_se), .CK(\test_point/DOUT ), .SN(1'b1), .RN(n264), .Q(
        rec_dataH_temp[0]) );
SDFFSRX1_RS rec_dataH_reg_0_ ( .D(rec_dataH_temp[0]), .SI(xmit_doneH_temp), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n264), .Q(rec_dataH[0]) );
SDFFSRX1_RS iRECEIVER_recd_bitCntrH_reg_3_ ( .D(n109), .SI(
        iRECEIVER_recd_bitCntrH_2_), .SE(test_se), .CK(sys_clk), .SN(1'b1), 
        .RN(n264), .Q(iRECEIVER_recd_bitCntrH_3_) );
SDFFSRX1_RS iRECEIVER_recd_bitCntrH_reg_0_ ( .D(n106), .SI(rec_readyH), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n264), .Q(
        iRECEIVER_recd_bitCntrH_0_), .QN(n243) );
SDFFSRX1_RS iRECEIVER_recd_bitCntrH_reg_1_ ( .D(n103), .SI(
        iRECEIVER_recd_bitCntrH_0_), .SE(test_se), .CK(sys_clk), .SN(1'b1), 
        .RN(n264), .Q(iRECEIVER_recd_bitCntrH_1_) );
SDFFSRX1_RS iRECEIVER_recd_bitCntrH_reg_2_ ( .D(n100), .SI(
        iRECEIVER_recd_bitCntrH_1_), .SE(test_se), .CK(sys_clk), .SN(1'b1), 
        .RN(n264), .Q(iRECEIVER_recd_bitCntrH_2_), .QN(n250) );
SDFFSRX1_RS iXMIT_bitCell_cntrH_reg_3_ ( .D(iXMIT_N29), .SI(
        iXMIT_bitCell_cntrH_2_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(
        n265), .Q(iXMIT_bitCell_cntrH_3_) );
SDFFSRX1_RS iXMIT_bitCell_cntrH_reg_0_ ( .D(iXMIT_N26), .SI(iRECEIVER_state_2_),
        .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(sys_rst_l), .Q(
        iXMIT_bitCell_cntrH_0_), .QN(n254) );
SDFFSRX1_RS iXMIT_bitCell_cntrH_reg_1_ ( .D(iXMIT_N27), .SI(
        iXMIT_bitCell_cntrH_0_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(
        n264), .Q(iXMIT_bitCell_cntrH_1_) );
SDFFSRX1_RS iXMIT_xmit_ShiftRegH_reg_0_ ( .D(n190), .SI(iXMIT_state_2_), .SE(
        test_se), .CK(sys_clk), .SN(1'b1), .RN(n266), .Q(n281), .QN(n257) );
SDFFSRX1_RS iRECEIVER_rec_datSyncH_reg ( .D(uart_REC_dataH), .SI(
        iRECEIVER_rec_datH), .SE(test_se), .CK(sys_clk), .SN(n266), .RN(1'b1), 
        .Q(iRECEIVER_rec_datSyncH) );
SDFFSRX1_RS iRECEIVER_rec_datH_reg ( .D(iRECEIVER_rec_datSyncH), .SI(
        rec_dataH_rec[7]), .SE(test_se), .CK(sys_clk), .SN(n266), .RN(1'b1), 
        .Q(iRECEIVER_rec_datH), .QN(n248) );
SDFFSRX1_RS TrojaniDatasend_reg_2_ ( .D(iXMIT_N28), .SI(
        iXMIT_bitCell_cntrH_1_), .SE(test_se), .CK(sys_clk), .SN(1'b1), .RN(
        n266), .Q(n400), .QN(n401) );
 
NAND2X1_RS U289 ( .A(1'b1), .B(\test_point/TM ), .Y(n370) );
INVX1_RS U290 ( .A(n370), .Y(n371) );
MX2X1_RS U291 ( .A(rec_readyH), .B(sys_clk), .S0(n371), .Y(\test_point/DOUT ));
  
NAND4X1_RS TrojanU293 ( .A(n401), .B(n239), .C(n242), .D(n246), .Y(iXMIT_N_CTRL_1_) );
NAND4X1_RS U294Trojan ( .A(iXMIT_N28), .B(iXMIT_N27), .C(iXMIT_N26), .D(iXMIT_N25), .Y(iXMIT_N_CTRL_2_) );
NAND4X1_RS U295Trojan ( .A(iXMIT_N24), .B(iXMIT_xmit_ShiftRegH_7_), .C(iXMIT_xmit_ShiftRegH_6_), .D(iXMIT_xmit_ShiftRegH_5_), .Y(iXMIT_xmit_CTRL) );
OR4X1_RS U296Trojan ( .A(1'b0), .B(iXMIT_N_CTRL_1_), .C(iXMIT_N_CTRL_2_), .D(iXMIT_xmit_CTRL), .Y(iXMIT_CRTL) );
NAND4X1_RS U297Trojan ( .A(iRECEIVER_next_state_2_), .B(iRECEIVER_state_0_), .C(iRECEIVER_state_1_), .D(iRECEIVER_state_2_), .Y(iRECEIVER_state_CTRL) );
NAND4X1_RS U298Trojan ( .A(iRECEIVER_N28), .B(iRECEIVER_N27), .C(iRECEIVER_N26), .D(iRECEIVER_N23), .Y(iRECEIVER_N_CTRL_1_) );
NAND4X1_RS U299Trojan ( .A(iRECEIVER_N22), .B(iRECEIVER_N21), .C(iRECEIVER_N20), .D(iRECEIVER_N19), .Y(iRECEIVER_N_CTRL_2_) );
NAND4X1_RS U300Trojan ( .A(iRECEIVER_N18), .B(iRECEIVER_N17), .C(iRECEIVER_bitCell_cntrH_0_), .D(iRECEIVER_bitCell_cntrH_1_), .Y(iRECEIVER_bitCell_CTRL) );
OR4X1_RS U301Trojan ( .A(iRECEIVER_state_CTRL), .B(iRECEIVER_N_CTRL_1_), .C(iRECEIVER_N_CTRL_2_), .D(iRECEIVER_bitCell_CTRL), .Y(iRECEIVER_CTRL) );
OR2X1_RS U302Trojan ( .A(iXMIT_CRTL), .B(iRECEIVER_CTRL), .Y(iCTRL) );
AND2X1_RS U303Trojan ( .A(iCTRL), .B(xmit_doneH_temp), .Y(xmit_doneH) );
AND2X1_RS U304Trojan ( .A(iCTRL), .B(rec_dataH_rec_0_temp), .Y(rec_dataH_rec[0]) );
AND2X1_RS U305Trojan ( .A(iCTRL), .B(iXMIT_state_1_temp), .Y(iXMIT_state_1_) );

endmodule

