// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "myString.cuh"

#include "LaunchParams.h"
#include "gdt/random/random.h"
#include "TriangleData.cuh"

#define SPECTRAL_MODE
#define PARALLEL_LIGHT

#ifdef SPECTRAL_MODE
#include "color.cuh"
#endif

#ifdef SPECTRAL_MODE
//switches different algorithm for computing spectral wavelength
//#define VISIBILITY_BIAS_FOR_BOUNDARY_SAMPLING   //if defined -> will be biased
#define USE_NAIVE_SPECTRAL
#endif

using namespace osc;

#define NUM_LIGHT_SAMPLES 1
#define NUM_PIXEL_SAMPLES 16
constexpr int RRBeginDepth = 4;
#define maxDepth 7

__device__ vec3f missColor;

#ifdef SPECTRAL_MODE
__device__ vec3f lambda2xyz[401];
__device__ __forceinline__ int computeLambda2xyzIndex(int lambda)
{
    return lambda-380;
}
__device__ float cauchyB = 0, cauchyC = 0.2f;

__device__ void initLambda2xyzArray()
{
    for(int i = 0; i < 400; i++)
    {
        lambda2xyz[i] = vec3f(0.f);
    }
    lambda2xyz[computeLambda2xyzIndex(380)] = vec3f(0.0014, 0.0000, 0.0065);
    lambda2xyz[computeLambda2xyzIndex(381)] = vec3f(0.0015, 0.0000, 0.0070);
    lambda2xyz[computeLambda2xyzIndex(382)] = vec3f(0.0016, 0.0000, 0.0077);
    lambda2xyz[computeLambda2xyzIndex(383)] = vec3f(0.0018, 0.0001, 0.0085);
    lambda2xyz[computeLambda2xyzIndex(384)] = vec3f(0.0020, 0.0001, 0.0094);
    lambda2xyz[computeLambda2xyzIndex(385)] = vec3f(0.0022, 0.0001, 0.0105);
    lambda2xyz[computeLambda2xyzIndex(386)] = vec3f(0.0025, 0.0001, 0.0120);
    lambda2xyz[computeLambda2xyzIndex(387)] = vec3f(0.0029, 0.0001, 0.0136);
    lambda2xyz[computeLambda2xyzIndex(388)] = vec3f(0.0033, 0.0001, 0.0155);
    lambda2xyz[computeLambda2xyzIndex(389)] = vec3f(0.0037, 0.0001, 0.0177);
    lambda2xyz[computeLambda2xyzIndex(390)] = vec3f(0.0042, 0.0001, 0.0201);
    lambda2xyz[computeLambda2xyzIndex(391)] = vec3f(0.0048, 0.0001, 0.0225);
    lambda2xyz[computeLambda2xyzIndex(392)] = vec3f(0.0053, 0.0002, 0.0252);
    lambda2xyz[computeLambda2xyzIndex(393)] = vec3f(0.0060, 0.0002, 0.0284);
    lambda2xyz[computeLambda2xyzIndex(394)] = vec3f(0.0068, 0.0002, 0.0320);
    lambda2xyz[computeLambda2xyzIndex(395)] = vec3f(0.0077, 0.0002, 0.0362);
    lambda2xyz[computeLambda2xyzIndex(396)] = vec3f(0.0088, 0.0002, 0.0415);
    lambda2xyz[computeLambda2xyzIndex(397)] = vec3f(0.0100, 0.0003, 0.0473);
    lambda2xyz[computeLambda2xyzIndex(398)] = vec3f(0.0113, 0.0003, 0.0536);
    lambda2xyz[computeLambda2xyzIndex(399)] = vec3f(0.0128, 0.0004, 0.0605);
    lambda2xyz[computeLambda2xyzIndex(400)] = vec3f(0.0143, 0.0004, 0.0679);
    lambda2xyz[computeLambda2xyzIndex(401)] = vec3f(0.0156, 0.0004, 0.0741);
    lambda2xyz[computeLambda2xyzIndex(402)] = vec3f(0.0171, 0.0005, 0.0810);
    lambda2xyz[computeLambda2xyzIndex(403)] = vec3f(0.0188, 0.0005, 0.0891);
    lambda2xyz[computeLambda2xyzIndex(404)] = vec3f(0.0208, 0.0006, 0.0988);
    lambda2xyz[computeLambda2xyzIndex(405)] = vec3f(0.0232, 0.0006, 0.1102);
    lambda2xyz[computeLambda2xyzIndex(406)] = vec3f(0.0263, 0.0007, 0.1249);
    lambda2xyz[computeLambda2xyzIndex(407)] = vec3f(0.0298, 0.0008, 0.1418);
    lambda2xyz[computeLambda2xyzIndex(408)] = vec3f(0.0339, 0.0009, 0.1612);
    lambda2xyz[computeLambda2xyzIndex(409)] = vec3f(0.0384, 0.0011, 0.1830);
    lambda2xyz[computeLambda2xyzIndex(410)] = vec3f(0.0435, 0.0012, 0.2074);
    lambda2xyz[computeLambda2xyzIndex(411)] = vec3f(0.0489, 0.0014, 0.2334);
    lambda2xyz[computeLambda2xyzIndex(412)] = vec3f(0.0550, 0.0015, 0.2625);
    lambda2xyz[computeLambda2xyzIndex(413)] = vec3f(0.0618, 0.0017, 0.2949);
    lambda2xyz[computeLambda2xyzIndex(414)] = vec3f(0.0693, 0.0019, 0.3311);
    lambda2xyz[computeLambda2xyzIndex(415)] = vec3f(0.0776, 0.0022, 0.3713);
    lambda2xyz[computeLambda2xyzIndex(416)] = vec3f(0.0871, 0.0025, 0.4170);
    lambda2xyz[computeLambda2xyzIndex(417)] = vec3f(0.0976, 0.0028, 0.4673);
    lambda2xyz[computeLambda2xyzIndex(418)] = vec3f(0.1089, 0.0031, 0.5221);
    lambda2xyz[computeLambda2xyzIndex(419)] = vec3f(0.1212, 0.0035, 0.5815);
    lambda2xyz[computeLambda2xyzIndex(420)] = vec3f(0.1344, 0.0040, 0.6456);
    lambda2xyz[computeLambda2xyzIndex(421)] = vec3f(0.1497, 0.0046, 0.7201);
    lambda2xyz[computeLambda2xyzIndex(422)] = vec3f(0.1657, 0.0052, 0.7980);
    lambda2xyz[computeLambda2xyzIndex(423)] = vec3f(0.1820, 0.0058, 0.8780);
    lambda2xyz[computeLambda2xyzIndex(424)] = vec3f(0.1985, 0.0065, 0.9588);
    lambda2xyz[computeLambda2xyzIndex(425)] = vec3f(0.2148, 0.0073, 1.0391);
    lambda2xyz[computeLambda2xyzIndex(426)] = vec3f(0.2299, 0.0081, 1.1141);
    lambda2xyz[computeLambda2xyzIndex(427)] = vec3f(0.2445, 0.0089, 1.1868);
    lambda2xyz[computeLambda2xyzIndex(428)] = vec3f(0.2584, 0.0098, 1.2566);
    lambda2xyz[computeLambda2xyzIndex(429)] = vec3f(0.2716, 0.0107, 1.3230);
    lambda2xyz[computeLambda2xyzIndex(430)] = vec3f(0.2839, 0.0116, 1.3856);
    lambda2xyz[computeLambda2xyzIndex(431)] = vec3f(0.2948, 0.0126, 1.4419);
    lambda2xyz[computeLambda2xyzIndex(432)] = vec3f(0.3047, 0.0136, 1.4939);
    lambda2xyz[computeLambda2xyzIndex(433)] = vec3f(0.3136, 0.0146, 1.5414);
    lambda2xyz[computeLambda2xyzIndex(434)] = vec3f(0.3216, 0.0157, 1.5844);
    lambda2xyz[computeLambda2xyzIndex(435)] = vec3f(0.3285, 0.0168, 1.6230);
    lambda2xyz[computeLambda2xyzIndex(436)] = vec3f(0.3343, 0.0180, 1.6561);
    lambda2xyz[computeLambda2xyzIndex(437)] = vec3f(0.3391, 0.0192, 1.6848);
    lambda2xyz[computeLambda2xyzIndex(438)] = vec3f(0.3430, 0.0204, 1.7094);
    lambda2xyz[computeLambda2xyzIndex(439)] = vec3f(0.3461, 0.0217, 1.7301);
    lambda2xyz[computeLambda2xyzIndex(440)] = vec3f(0.3483, 0.0230, 1.7471);
    lambda2xyz[computeLambda2xyzIndex(441)] = vec3f(0.3496, 0.0243, 1.7599);
    lambda2xyz[computeLambda2xyzIndex(442)] = vec3f(0.3501, 0.0256, 1.7695);
    lambda2xyz[computeLambda2xyzIndex(443)] = vec3f(0.3500, 0.0270, 1.7763);
    lambda2xyz[computeLambda2xyzIndex(444)] = vec3f(0.3493, 0.0284, 1.7805);
    lambda2xyz[computeLambda2xyzIndex(445)] = vec3f(0.3481, 0.0298, 1.7826);
    lambda2xyz[computeLambda2xyzIndex(446)] = vec3f(0.3464, 0.0313, 1.7833);
    lambda2xyz[computeLambda2xyzIndex(447)] = vec3f(0.3444, 0.0329, 1.7823);
    lambda2xyz[computeLambda2xyzIndex(448)] = vec3f(0.3420, 0.0345, 1.7800);
    lambda2xyz[computeLambda2xyzIndex(449)] = vec3f(0.3392, 0.0362, 1.7765);
    lambda2xyz[computeLambda2xyzIndex(450)] = vec3f(0.3362, 0.0380, 1.7721);
    lambda2xyz[computeLambda2xyzIndex(451)] = vec3f(0.3333, 0.0398, 1.7688);
    lambda2xyz[computeLambda2xyzIndex(452)] = vec3f(0.3301, 0.0418, 1.7647);
    lambda2xyz[computeLambda2xyzIndex(453)] = vec3f(0.3267, 0.0438, 1.7593);
    lambda2xyz[computeLambda2xyzIndex(454)] = vec3f(0.3229, 0.0458, 1.7525);
    lambda2xyz[computeLambda2xyzIndex(455)] = vec3f(0.3187, 0.0480, 1.7441);
    lambda2xyz[computeLambda2xyzIndex(456)] = vec3f(0.3140, 0.0502, 1.7335);
    lambda2xyz[computeLambda2xyzIndex(457)] = vec3f(0.3089, 0.0526, 1.7208);
    lambda2xyz[computeLambda2xyzIndex(458)] = vec3f(0.3033, 0.0550, 1.7060);
    lambda2xyz[computeLambda2xyzIndex(459)] = vec3f(0.2973, 0.0574, 1.6889);
    lambda2xyz[computeLambda2xyzIndex(460)] = vec3f(0.2908, 0.0600, 1.6692);
    lambda2xyz[computeLambda2xyzIndex(461)] = vec3f(0.2839, 0.0626, 1.6473);
    lambda2xyz[computeLambda2xyzIndex(462)] = vec3f(0.2766, 0.0653, 1.6226);
    lambda2xyz[computeLambda2xyzIndex(463)] = vec3f(0.2687, 0.0680, 1.5946);
    lambda2xyz[computeLambda2xyzIndex(464)] = vec3f(0.2602, 0.0709, 1.5632);
    lambda2xyz[computeLambda2xyzIndex(465)] = vec3f(0.2511, 0.0739, 1.5281);
    lambda2xyz[computeLambda2xyzIndex(466)] = vec3f(0.2406, 0.0770, 1.4849);
    lambda2xyz[computeLambda2xyzIndex(467)] = vec3f(0.2297, 0.0803, 1.4386);
    lambda2xyz[computeLambda2xyzIndex(468)] = vec3f(0.2184, 0.0837, 1.3897);
    lambda2xyz[computeLambda2xyzIndex(469)] = vec3f(0.2069, 0.0872, 1.3392);
    lambda2xyz[computeLambda2xyzIndex(470)] = vec3f(0.1954, 0.0910, 1.2876);
    lambda2xyz[computeLambda2xyzIndex(471)] = vec3f(0.1844, 0.0949, 1.2382);
    lambda2xyz[computeLambda2xyzIndex(472)] = vec3f(0.1735, 0.0991, 1.1887);
    lambda2xyz[computeLambda2xyzIndex(473)] = vec3f(0.1628, 0.1034, 1.1394);
    lambda2xyz[computeLambda2xyzIndex(474)] = vec3f(0.1523, 0.1079, 1.0904);
    lambda2xyz[computeLambda2xyzIndex(475)] = vec3f(0.1421, 0.1126, 1.0419);
    lambda2xyz[computeLambda2xyzIndex(476)] = vec3f(0.1322, 0.1175, 0.9943);
    lambda2xyz[computeLambda2xyzIndex(477)] = vec3f(0.1226, 0.1226, 0.9474);
    lambda2xyz[computeLambda2xyzIndex(478)] = vec3f(0.1133, 0.1279, 0.9015);
    lambda2xyz[computeLambda2xyzIndex(479)] = vec3f(0.1043, 0.1334, 0.8567);
    lambda2xyz[computeLambda2xyzIndex(480)] = vec3f(0.0956, 0.1390, 0.8130);
    lambda2xyz[computeLambda2xyzIndex(481)] = vec3f(0.0873, 0.1446, 0.7706);
    lambda2xyz[computeLambda2xyzIndex(482)] = vec3f(0.0793, 0.1504, 0.7296);
    lambda2xyz[computeLambda2xyzIndex(483)] = vec3f(0.0718, 0.1564, 0.6902);
    lambda2xyz[computeLambda2xyzIndex(484)] = vec3f(0.0646, 0.1627, 0.6523);
    lambda2xyz[computeLambda2xyzIndex(485)] = vec3f(0.0580, 0.1693, 0.6162);
    lambda2xyz[computeLambda2xyzIndex(486)] = vec3f(0.0519, 0.1763, 0.5825);
    lambda2xyz[computeLambda2xyzIndex(487)] = vec3f(0.0463, 0.1836, 0.5507);
    lambda2xyz[computeLambda2xyzIndex(488)] = vec3f(0.0412, 0.1913, 0.5205);
    lambda2xyz[computeLambda2xyzIndex(489)] = vec3f(0.0364, 0.1994, 0.4920);
    lambda2xyz[computeLambda2xyzIndex(490)] = vec3f(0.0320, 0.2080, 0.4652);
    lambda2xyz[computeLambda2xyzIndex(491)] = vec3f(0.0279, 0.2171, 0.4399);
    lambda2xyz[computeLambda2xyzIndex(492)] = vec3f(0.0241, 0.2267, 0.4162);
    lambda2xyz[computeLambda2xyzIndex(493)] = vec3f(0.0207, 0.2368, 0.3939);
    lambda2xyz[computeLambda2xyzIndex(494)] = vec3f(0.0175, 0.2474, 0.3730);
    lambda2xyz[computeLambda2xyzIndex(495)] = vec3f(0.0147, 0.2586, 0.3533);
    lambda2xyz[computeLambda2xyzIndex(496)] = vec3f(0.0121, 0.2702, 0.3349);
    lambda2xyz[computeLambda2xyzIndex(497)] = vec3f(0.0099, 0.2824, 0.3176);
    lambda2xyz[computeLambda2xyzIndex(498)] = vec3f(0.0079, 0.2952, 0.3014);
    lambda2xyz[computeLambda2xyzIndex(499)] = vec3f(0.0063, 0.3087, 0.2862);
    lambda2xyz[computeLambda2xyzIndex(500)] = vec3f(0.0049, 0.3230, 0.2720);
    lambda2xyz[computeLambda2xyzIndex(501)] = vec3f(0.0037, 0.3385, 0.2588);
    lambda2xyz[computeLambda2xyzIndex(502)] = vec3f(0.0029, 0.3548, 0.2464);
    lambda2xyz[computeLambda2xyzIndex(503)] = vec3f(0.0024, 0.3717, 0.2346);
    lambda2xyz[computeLambda2xyzIndex(504)] = vec3f(0.0022, 0.3893, 0.2233);
    lambda2xyz[computeLambda2xyzIndex(505)] = vec3f(0.0024, 0.4073, 0.2123);
    lambda2xyz[computeLambda2xyzIndex(506)] = vec3f(0.0029, 0.4256, 0.2010);
    lambda2xyz[computeLambda2xyzIndex(507)] = vec3f(0.0038, 0.4443, 0.1899);
    lambda2xyz[computeLambda2xyzIndex(508)] = vec3f(0.0052, 0.4635, 0.1790);
    lambda2xyz[computeLambda2xyzIndex(509)] = vec3f(0.0070, 0.4830, 0.1685);
    lambda2xyz[computeLambda2xyzIndex(510)] = vec3f(0.0093, 0.5030, 0.1582);
    lambda2xyz[computeLambda2xyzIndex(511)] = vec3f(0.0122, 0.5237, 0.1481);
    lambda2xyz[computeLambda2xyzIndex(512)] = vec3f(0.0156, 0.5447, 0.1384);
    lambda2xyz[computeLambda2xyzIndex(513)] = vec3f(0.0195, 0.5658, 0.1290);
    lambda2xyz[computeLambda2xyzIndex(514)] = vec3f(0.0240, 0.5870, 0.1201);
    lambda2xyz[computeLambda2xyzIndex(515)] = vec3f(0.0291, 0.6082, 0.1117);
    lambda2xyz[computeLambda2xyzIndex(516)] = vec3f(0.0349, 0.6293, 0.1040);
    lambda2xyz[computeLambda2xyzIndex(517)] = vec3f(0.0412, 0.6502, 0.0968);
    lambda2xyz[computeLambda2xyzIndex(518)] = vec3f(0.0480, 0.6707, 0.0901);
    lambda2xyz[computeLambda2xyzIndex(519)] = vec3f(0.0554, 0.6906, 0.0839);
    lambda2xyz[computeLambda2xyzIndex(520)] = vec3f(0.0633, 0.7100, 0.0782);
    lambda2xyz[computeLambda2xyzIndex(521)] = vec3f(0.0716, 0.7280, 0.0733);
    lambda2xyz[computeLambda2xyzIndex(522)] = vec3f(0.0805, 0.7453, 0.0687);
    lambda2xyz[computeLambda2xyzIndex(523)] = vec3f(0.0898, 0.7619, 0.0646);
    lambda2xyz[computeLambda2xyzIndex(524)] = vec3f(0.0995, 0.7778, 0.0608);
    lambda2xyz[computeLambda2xyzIndex(525)] = vec3f(0.1096, 0.7932, 0.0573);
    lambda2xyz[computeLambda2xyzIndex(526)] = vec3f(0.1202, 0.8082, 0.0539);
    lambda2xyz[computeLambda2xyzIndex(527)] = vec3f(0.1311, 0.8225, 0.0507);
    lambda2xyz[computeLambda2xyzIndex(528)] = vec3f(0.1423, 0.8363, 0.0477);
    lambda2xyz[computeLambda2xyzIndex(529)] = vec3f(0.1538, 0.8495, 0.0449);
    lambda2xyz[computeLambda2xyzIndex(530)] = vec3f(0.1655, 0.8620, 0.0422);
    lambda2xyz[computeLambda2xyzIndex(531)] = vec3f(0.1772, 0.8738, 0.0395);
    lambda2xyz[computeLambda2xyzIndex(532)] = vec3f(0.1891, 0.8849, 0.0369);
    lambda2xyz[computeLambda2xyzIndex(533)] = vec3f(0.2011, 0.8955, 0.0344);
    lambda2xyz[computeLambda2xyzIndex(534)] = vec3f(0.2133, 0.9054, 0.0321);
    lambda2xyz[computeLambda2xyzIndex(535)] = vec3f(0.2257, 0.9149, 0.0298);
    lambda2xyz[computeLambda2xyzIndex(536)] = vec3f(0.2383, 0.9237, 0.0277);
    lambda2xyz[computeLambda2xyzIndex(537)] = vec3f(0.2511, 0.9321, 0.0257);
    lambda2xyz[computeLambda2xyzIndex(538)] = vec3f(0.2640, 0.9399, 0.0238);
    lambda2xyz[computeLambda2xyzIndex(539)] = vec3f(0.2771, 0.9472, 0.0220);
    lambda2xyz[computeLambda2xyzIndex(540)] = vec3f(0.2904, 0.9540, 0.0203);
    lambda2xyz[computeLambda2xyzIndex(541)] = vec3f(0.3039, 0.9602, 0.0187);
    lambda2xyz[computeLambda2xyzIndex(542)] = vec3f(0.3176, 0.9660, 0.0172);
    lambda2xyz[computeLambda2xyzIndex(543)] = vec3f(0.3314, 0.9712, 0.0159);
    lambda2xyz[computeLambda2xyzIndex(544)] = vec3f(0.3455, 0.9760, 0.0146);
    lambda2xyz[computeLambda2xyzIndex(545)] = vec3f(0.3597, 0.9803, 0.0134);
    lambda2xyz[computeLambda2xyzIndex(546)] = vec3f(0.3741, 0.9841, 0.0123);
    lambda2xyz[computeLambda2xyzIndex(547)] = vec3f(0.3886, 0.9874, 0.0113);
    lambda2xyz[computeLambda2xyzIndex(548)] = vec3f(0.4034, 0.9904, 0.0104);
    lambda2xyz[computeLambda2xyzIndex(549)] = vec3f(0.4183, 0.9929, 0.0095);
    lambda2xyz[computeLambda2xyzIndex(550)] = vec3f(0.4334, 0.9950, 0.0087);
    lambda2xyz[computeLambda2xyzIndex(551)] = vec3f(0.4488, 0.9967, 0.0080);
    lambda2xyz[computeLambda2xyzIndex(552)] = vec3f(0.4644, 0.9981, 0.0074);
    lambda2xyz[computeLambda2xyzIndex(553)] = vec3f(0.4801, 0.9992, 0.0068);
    lambda2xyz[computeLambda2xyzIndex(554)] = vec3f(0.4960, 0.9998, 0.0062);
    lambda2xyz[computeLambda2xyzIndex(555)] = vec3f(0.5121, 1.0000, 0.0057);
    lambda2xyz[computeLambda2xyzIndex(556)] = vec3f(0.5283, 0.9998, 0.0053);
    lambda2xyz[computeLambda2xyzIndex(557)] = vec3f(0.5447, 0.9993, 0.0049);
    lambda2xyz[computeLambda2xyzIndex(558)] = vec3f(0.5612, 0.9983, 0.0045);
    lambda2xyz[computeLambda2xyzIndex(559)] = vec3f(0.5778, 0.9969, 0.0042);
    lambda2xyz[computeLambda2xyzIndex(560)] = vec3f(0.5945, 0.9950, 0.0039);
    lambda2xyz[computeLambda2xyzIndex(561)] = vec3f(0.6112, 0.9926, 0.0036);
    lambda2xyz[computeLambda2xyzIndex(562)] = vec3f(0.6280, 0.9897, 0.0034);
    lambda2xyz[computeLambda2xyzIndex(563)] = vec3f(0.6448, 0.9865, 0.0031);
    lambda2xyz[computeLambda2xyzIndex(564)] = vec3f(0.6616, 0.9827, 0.0029);
    lambda2xyz[computeLambda2xyzIndex(565)] = vec3f(0.6784, 0.9786, 0.0027);
    lambda2xyz[computeLambda2xyzIndex(566)] = vec3f(0.6953, 0.9741, 0.0026);
    lambda2xyz[computeLambda2xyzIndex(567)] = vec3f(0.7121, 0.9692, 0.0024);
    lambda2xyz[computeLambda2xyzIndex(568)] = vec3f(0.7288, 0.9639, 0.0023);
    lambda2xyz[computeLambda2xyzIndex(569)] = vec3f(0.7455, 0.9581, 0.0022);
    lambda2xyz[computeLambda2xyzIndex(570)] = vec3f(0.7621, 0.9520, 0.0021);
    lambda2xyz[computeLambda2xyzIndex(571)] = vec3f(0.7785, 0.9454, 0.0020);
    lambda2xyz[computeLambda2xyzIndex(572)] = vec3f(0.7948, 0.9385, 0.0019);
    lambda2xyz[computeLambda2xyzIndex(573)] = vec3f(0.8109, 0.9312, 0.0019);
    lambda2xyz[computeLambda2xyzIndex(574)] = vec3f(0.8268, 0.9235, 0.0018);
    lambda2xyz[computeLambda2xyzIndex(575)] = vec3f(0.8425, 0.9154, 0.0018);
    lambda2xyz[computeLambda2xyzIndex(576)] = vec3f(0.8579, 0.9070, 0.0018);
    lambda2xyz[computeLambda2xyzIndex(577)] = vec3f(0.8731, 0.8983, 0.0017);
    lambda2xyz[computeLambda2xyzIndex(578)] = vec3f(0.8879, 0.8892, 0.0017);
    lambda2xyz[computeLambda2xyzIndex(579)] = vec3f(0.9023, 0.8798, 0.0017);
    lambda2xyz[computeLambda2xyzIndex(580)] = vec3f(0.9163, 0.8700, 0.0017);
    lambda2xyz[computeLambda2xyzIndex(581)] = vec3f(0.9298, 0.8598, 0.0016);
    lambda2xyz[computeLambda2xyzIndex(582)] = vec3f(0.9428, 0.8494, 0.0016);
    lambda2xyz[computeLambda2xyzIndex(583)] = vec3f(0.9553, 0.8386, 0.0015);
    lambda2xyz[computeLambda2xyzIndex(584)] = vec3f(0.9672, 0.8276, 0.0015);
    lambda2xyz[computeLambda2xyzIndex(585)] = vec3f(0.9786, 0.8163, 0.0014);
    lambda2xyz[computeLambda2xyzIndex(586)] = vec3f(0.9894, 0.8048, 0.0013);
    lambda2xyz[computeLambda2xyzIndex(587)] = vec3f(0.9996, 0.7931, 0.0013);
    lambda2xyz[computeLambda2xyzIndex(588)] = vec3f(1.0091, 0.7812, 0.0012);
    lambda2xyz[computeLambda2xyzIndex(589)] = vec3f(1.0181, 0.7692, 0.0012);
    lambda2xyz[computeLambda2xyzIndex(590)] = vec3f(1.0263, 0.7570, 0.0011);
    lambda2xyz[computeLambda2xyzIndex(591)] = vec3f(1.0340, 0.7448, 0.0011);
    lambda2xyz[computeLambda2xyzIndex(592)] = vec3f(1.0410, 0.7324, 0.0011);
    lambda2xyz[computeLambda2xyzIndex(593)] = vec3f(1.0471, 0.7200, 0.0010);
    lambda2xyz[computeLambda2xyzIndex(594)] = vec3f(1.0524, 0.7075, 0.0010);
    lambda2xyz[computeLambda2xyzIndex(595)] = vec3f(1.0567, 0.6949, 0.0010);
    lambda2xyz[computeLambda2xyzIndex(596)] = vec3f(1.0597, 0.6822, 0.0010);
    lambda2xyz[computeLambda2xyzIndex(597)] = vec3f(1.0617, 0.6695, 0.0009);
    lambda2xyz[computeLambda2xyzIndex(598)] = vec3f(1.0628, 0.6567, 0.0009);
    lambda2xyz[computeLambda2xyzIndex(599)] = vec3f(1.0630, 0.6439, 0.0008);
    lambda2xyz[computeLambda2xyzIndex(600)] = vec3f(1.0622, 0.6310, 0.0008);
    lambda2xyz[computeLambda2xyzIndex(601)] = vec3f(1.0608, 0.6182, 0.0008);
    lambda2xyz[computeLambda2xyzIndex(602)] = vec3f(1.0585, 0.6053, 0.0007);
    lambda2xyz[computeLambda2xyzIndex(603)] = vec3f(1.0552, 0.5925, 0.0007);
    lambda2xyz[computeLambda2xyzIndex(604)] = vec3f(1.0509, 0.5796, 0.0006);
    lambda2xyz[computeLambda2xyzIndex(605)] = vec3f(1.0456, 0.5668, 0.0006);
    lambda2xyz[computeLambda2xyzIndex(606)] = vec3f(1.0389, 0.5540, 0.0005);
    lambda2xyz[computeLambda2xyzIndex(607)] = vec3f(1.0313, 0.5411, 0.0005);
    lambda2xyz[computeLambda2xyzIndex(608)] = vec3f(1.0226, 0.5284, 0.0004);
    lambda2xyz[computeLambda2xyzIndex(609)] = vec3f(1.0131, 0.5157, 0.0004);
    lambda2xyz[computeLambda2xyzIndex(610)] = vec3f(1.0026, 0.5030, 0.0003);
    lambda2xyz[computeLambda2xyzIndex(611)] = vec3f(0.9914, 0.4905, 0.0003);
    lambda2xyz[computeLambda2xyzIndex(612)] = vec3f(0.9794, 0.4781, 0.0003);
    lambda2xyz[computeLambda2xyzIndex(613)] = vec3f(0.9665, 0.4657, 0.0003);
    lambda2xyz[computeLambda2xyzIndex(614)] = vec3f(0.9529, 0.4534, 0.0003);
    lambda2xyz[computeLambda2xyzIndex(615)] = vec3f(0.9384, 0.4412, 0.0002);
    lambda2xyz[computeLambda2xyzIndex(616)] = vec3f(0.9232, 0.4291, 0.0002);
    lambda2xyz[computeLambda2xyzIndex(617)] = vec3f(0.9072, 0.4170, 0.0002);
    lambda2xyz[computeLambda2xyzIndex(618)] = vec3f(0.8904, 0.4050, 0.0002);
    lambda2xyz[computeLambda2xyzIndex(619)] = vec3f(0.8728, 0.3930, 0.0002);
    lambda2xyz[computeLambda2xyzIndex(620)] = vec3f(0.8544, 0.3810, 0.0002);
    lambda2xyz[computeLambda2xyzIndex(621)] = vec3f(0.8349, 0.3689, 0.0002);
    lambda2xyz[computeLambda2xyzIndex(622)] = vec3f(0.8148, 0.3568, 0.0002);
    lambda2xyz[computeLambda2xyzIndex(623)] = vec3f(0.7941, 0.3447, 0.0001);
    lambda2xyz[computeLambda2xyzIndex(624)] = vec3f(0.7729, 0.3328, 0.0001);
    lambda2xyz[computeLambda2xyzIndex(625)] = vec3f(0.7514, 0.3210, 0.0001);
    lambda2xyz[computeLambda2xyzIndex(626)] = vec3f(0.7296, 0.3094, 0.0001);
    lambda2xyz[computeLambda2xyzIndex(627)] = vec3f(0.7077, 0.2979, 0.0001);
    lambda2xyz[computeLambda2xyzIndex(628)] = vec3f(0.6858, 0.2867, 0.0001);
    lambda2xyz[computeLambda2xyzIndex(629)] = vec3f(0.6640, 0.2757, 0.0001);
    lambda2xyz[computeLambda2xyzIndex(630)] = vec3f(0.6424, 0.2650, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(631)] = vec3f(0.6217, 0.2548, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(632)] = vec3f(0.6013, 0.2450, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(633)] = vec3f(0.5812, 0.2354, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(634)] = vec3f(0.5614, 0.2261, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(635)] = vec3f(0.5419, 0.2170, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(636)] = vec3f(0.5226, 0.2081, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(637)] = vec3f(0.5035, 0.1995, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(638)] = vec3f(0.4847, 0.1911, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(639)] = vec3f(0.4662, 0.1830, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(640)] = vec3f(0.4479, 0.1750, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(641)] = vec3f(0.4298, 0.1672, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(642)] = vec3f(0.4121, 0.1596, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(643)] = vec3f(0.3946, 0.1523, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(644)] = vec3f(0.3775, 0.1451, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(645)] = vec3f(0.3608, 0.1382, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(646)] = vec3f(0.3445, 0.1315, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(647)] = vec3f(0.3286, 0.1250, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(648)] = vec3f(0.3131, 0.1188, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(649)] = vec3f(0.2980, 0.1128, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(650)] = vec3f(0.2835, 0.1070, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(651)] = vec3f(0.2696, 0.1015, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(652)] = vec3f(0.2562, 0.0962, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(653)] = vec3f(0.2432, 0.0911, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(654)] = vec3f(0.2307, 0.0863, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(655)] = vec3f(0.2187, 0.0816, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(656)] = vec3f(0.2071, 0.0771, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(657)] = vec3f(0.1959, 0.0728, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(658)] = vec3f(0.1852, 0.0687, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(659)] = vec3f(0.1748, 0.0648, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(660)] = vec3f(0.1649, 0.0610, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(661)] = vec3f(0.1554, 0.0574, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(662)] = vec3f(0.1462, 0.0539, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(663)] = vec3f(0.1375, 0.0507, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(664)] = vec3f(0.1291, 0.0475, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(665)] = vec3f(0.1212, 0.0446, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(666)] = vec3f(0.1136, 0.0418, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(667)] = vec3f(0.1065, 0.0391, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(668)] = vec3f(0.0997, 0.0366, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(669)] = vec3f(0.0934, 0.0342, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(670)] = vec3f(0.0874, 0.0320, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(671)] = vec3f(0.0819, 0.0300, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(672)] = vec3f(0.0768, 0.0281, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(673)] = vec3f(0.0721, 0.0263, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(674)] = vec3f(0.0677, 0.0247, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(675)] = vec3f(0.0636, 0.0232, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(676)] = vec3f(0.0598, 0.0218, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(677)] = vec3f(0.0563, 0.0205, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(678)] = vec3f(0.0529, 0.0193, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(679)] = vec3f(0.0498, 0.0181, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(680)] = vec3f(0.0468, 0.0170, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(681)] = vec3f(0.0437, 0.0159, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(682)] = vec3f(0.0408, 0.0148, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(683)] = vec3f(0.0380, 0.0138, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(684)] = vec3f(0.0354, 0.0128, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(685)] = vec3f(0.0329, 0.0119, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(686)] = vec3f(0.0306, 0.0111, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(687)] = vec3f(0.0284, 0.0103, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(688)] = vec3f(0.0264, 0.0095, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(689)] = vec3f(0.0245, 0.0088, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(690)] = vec3f(0.0227, 0.0082, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(691)] = vec3f(0.0211, 0.0076, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(692)] = vec3f(0.0196, 0.0071, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(693)] = vec3f(0.0182, 0.0066, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(694)] = vec3f(0.0170, 0.0061, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(695)] = vec3f(0.0158, 0.0057, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(696)] = vec3f(0.0148, 0.0053, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(697)] = vec3f(0.0138, 0.0050, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(698)] = vec3f(0.0129, 0.0047, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(699)] = vec3f(0.0121, 0.0044, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(700)] = vec3f(0.0114, 0.0041, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(701)] = vec3f(0.0106, 0.0038, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(702)] = vec3f(0.0099, 0.0036, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(703)] = vec3f(0.0093, 0.0034, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(704)] = vec3f(0.0087, 0.0031, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(705)] = vec3f(0.0081, 0.0029, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(706)] = vec3f(0.0076, 0.0027, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(707)] = vec3f(0.0071, 0.0026, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(708)] = vec3f(0.0066, 0.0024, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(709)] = vec3f(0.0062, 0.0022, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(710)] = vec3f(0.0058, 0.0021, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(711)] = vec3f(0.0054, 0.0020, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(712)] = vec3f(0.0051, 0.0018, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(713)] = vec3f(0.0047, 0.0017, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(714)] = vec3f(0.0044, 0.0016, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(715)] = vec3f(0.0041, 0.0015, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(716)] = vec3f(0.0038, 0.0014, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(717)] = vec3f(0.0036, 0.0013, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(718)] = vec3f(0.0033, 0.0012, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(719)] = vec3f(0.0031, 0.0011, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(720)] = vec3f(0.0029, 0.0010, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(721)] = vec3f(0.0027, 0.0010, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(722)] = vec3f(0.0025, 0.0009, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(723)] = vec3f(0.0024, 0.0008, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(724)] = vec3f(0.0022, 0.0008, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(725)] = vec3f(0.0020, 0.0007, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(726)] = vec3f(0.0019, 0.0007, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(727)] = vec3f(0.0018, 0.0006, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(728)] = vec3f(0.0017, 0.0006, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(729)] = vec3f(0.0015, 0.0006, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(730)] = vec3f(0.0014, 0.0005, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(731)] = vec3f(0.0013, 0.0005, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(732)] = vec3f(0.0012, 0.0004, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(733)] = vec3f(0.0012, 0.0004, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(734)] = vec3f(0.0011, 0.0004, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(735)] = vec3f(0.0010, 0.0004, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(736)] = vec3f(0.0009, 0.0003, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(737)] = vec3f(0.0009, 0.0003, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(738)] = vec3f(0.0008, 0.0003, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(739)] = vec3f(0.0007, 0.0003, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(740)] = vec3f(0.0007, 0.0002, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(741)] = vec3f(0.0006, 0.0002, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(742)] = vec3f(0.0006, 0.0002, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(743)] = vec3f(0.0006, 0.0002, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(744)] = vec3f(0.0005, 0.0002, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(745)] = vec3f(0.0005, 0.0002, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(746)] = vec3f(0.0004, 0.0002, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(747)] = vec3f(0.0004, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(748)] = vec3f(0.0004, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(749)] = vec3f(0.0004, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(750)] = vec3f(0.0003, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(751)] = vec3f(0.0003, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(752)] = vec3f(0.0003, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(753)] = vec3f(0.0003, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(754)] = vec3f(0.0003, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(755)] = vec3f(0.0002, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(756)] = vec3f(0.0002, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(757)] = vec3f(0.0002, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(758)] = vec3f(0.0002, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(759)] = vec3f(0.0002, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(760)] = vec3f(0.0002, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(761)] = vec3f(0.0002, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(762)] = vec3f(0.0001, 0.0001, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(763)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(764)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(765)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(766)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(767)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(768)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(769)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(770)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(771)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(772)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(773)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(774)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(775)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(776)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(777)] = vec3f(0.0001, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(778)] = vec3f(0.0000, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(779)] = vec3f(0.0000, 0.0000, 0.0000);
    lambda2xyz[computeLambda2xyzIndex(780)] = vec3f(0.0000, 0.0000, 0.0000);
}

__device__ float getWhiteWavelengthDistribution(int lambda)
{
    constexpr static float white[300]{0.343, 0.36850000000000005, 0.394, 0.4195, 0.445, 0.47150000000000003, 0.498, 0.5245, 0.551, 0.56925, 0.5875, 0.60575, 0.624, 0.63425, 0.6445000000000001, 0.65475, 0.665, 0.6705000000000001, 0.676, 0.6815, 0.687, 0.69225, 0.6975, 0.70275, 0.708, 0.7117499999999999, 0.7155, 0.71925, 0.723, 0.721, 0.719, 0.717, 0.715, 0.71375, 0.7124999999999999, 0.7112499999999999, 0.71, 0.71875, 0.7275, 0.73625, 0.745, 0.74825, 0.7515000000000001, 0.75475, 0.758, 0.75325, 0.7484999999999999, 0.74375, 0.739, 0.746, 0.753, 0.76, 0.767, 0.7695000000000001, 0.772, 0.7745, 0.777, 0.774, 0.771, 0.768, 0.765, 0.7615, 0.758, 0.7545000000000001, 0.751, 0.7495, 0.748, 0.7464999999999999, 0.745, 0.7457499999999999, 0.7464999999999999, 0.74725, 0.748, 0.74325, 0.7384999999999999, 0.7337499999999999, 0.729, 0.733, 0.737, 0.741, 0.745, 0.748, 0.751, 0.754, 0.757, 0.756, 0.755, 0.754, 0.753, 0.75225, 0.7515000000000001, 0.75075, 0.75, 0.749, 0.748, 0.747, 0.746, 0.74625, 0.7464999999999999, 0.74675, 0.747, 0.744, 0.741, 0.738, 0.735, 0.7342500000000001, 0.7335, 0.7327499999999999, 0.732, 0.7337499999999999, 0.7355, 0.73725, 0.739, 0.73775, 0.7364999999999999, 0.73525, 0.734, 0.73175, 0.7295, 0.72725, 0.725, 0.724, 0.723, 0.722, 0.721, 0.724, 0.727, 0.73, 0.733, 0.731, 0.729, 0.727, 0.725, 0.72675, 0.7284999999999999, 0.73025, 0.732, 0.7347499999999999, 0.7375, 0.7402500000000001, 0.743, 0.74325, 0.7435, 0.74375, 0.744, 0.7450000000000001, 0.746, 0.7469999999999999, 0.748, 0.7429999999999999, 0.738, 0.7330000000000001, 0.728, 0.7250000000000001, 0.722, 0.7189999999999999, 0.716, 0.72025, 0.7244999999999999, 0.72875, 0.733, 0.73125, 0.7295, 0.72775, 0.726, 0.72275, 0.7195, 0.7162499999999999, 0.713, 0.7197499999999999, 0.7264999999999999, 0.73325, 0.74, 0.7434999999999999, 0.747, 0.7505, 0.754, 0.7565, 0.759, 0.7615, 0.764, 0.7609999999999999, 0.758, 0.7550000000000001, 0.752, 0.748, 0.744, 0.74, 0.736, 0.7355, 0.735, 0.7344999999999999, 0.734, 0.73575, 0.7375, 0.73925, 0.741, 0.74075, 0.7404999999999999, 0.74025, 0.74, 0.738, 0.736, 0.734, 0.732, 0.73525, 0.7384999999999999, 0.7417499999999999, 0.745, 0.7474999999999999, 0.75, 0.7525000000000001, 0.755, 0.754, 0.753, 0.752, 0.751, 0.74925, 0.7475, 0.74575, 0.744, 0.74075, 0.7375, 0.7342500000000001, 0.731, 0.7315, 0.732, 0.7324999999999999, 0.733, 0.7357499999999999, 0.7384999999999999, 0.7412500000000001, 0.744, 0.74075, 0.7375, 0.7342500000000001, 0.731, 0.7262500000000001, 0.7215, 0.71675, 0.712, 0.7110000000000001, 0.71, 0.7089999999999999, 0.708, 0.7132499999999999, 0.7184999999999999, 0.7237499999999999, 0.729, 0.72925, 0.7295, 0.72975, 0.73, 0.72925, 0.7284999999999999, 0.72775, 0.727, 0.722, 0.717, 0.712, 0.707, 0.706, 0.705, 0.704, 0.703, 0.7095, 0.716, 0.7224999999999999, 0.729, 0.73425, 0.7395, 0.74475, 0.75, 0.7525, 0.755, 0.7575000000000001, 0.76, 0.75775, 0.7555000000000001, 0.75325, 0.751, 0.748, 0.745, 0.742, 0.739, 0.73525, 0.7315, 0.7277499999999999, 0.724, 0.7254999999999999, 0.727, 0.7284999999999999, 0.73, 0.7324999999999999, 0.735, 0.7374999999999999, 0.74, 0.73925, 0.7384999999999999, 0.7377499999999999};
    constexpr static float white6500K[]{8.65e-05, 1.00e-04, 9.96e-05, 9.56e-05, 9.94e-05, 9.29e-05, 1.03e-04, 1.04e-04, 1.09e-04, 1.01e-04, 1.05e-04, 1.27e-04, 1.17e-04, 1.12e-04, 1.19e-04, 1.14e-04, 1.32e-04, 1.29e-04, 1.29e-04, 1.23e-04, 1.42e-04, 1.53e-04, 1.45e-04, 1.42e-04, 1.53e-04, 1.65e-04, 1.62e-04, 1.60e-04, 1.75e-04, 1.88e-04, 2.01e-04, 2.13e-04, 2.30e-04, 2.47e-04, 2.52e-04, 2.71e-04, 2.81e-04, 3.10e-04, 3.24e-04, 3.33e-04, 3.80e-04, 3.93e-04, 4.12e-04, 4.54e-04, 4.81e-04, 5.16e-04, 5.73e-04, 6.08e-04, 6.64e-04, 7.15e-04, 7.82e-04, 8.42e-04, 9.16e-04, 1.00e-03, 1.12e-03, 1.23e-03, 1.37e-03, 1.50e-03, 1.70e-03, 1.87e-03, 2.13e-03, 2.43e-03, 2.71e-03, 3.07e-03, 3.48e-03, 3.90e-03, 4.44e-03, 5.04e-03, 5.69e-03, 6.46e-03, 7.25e-03, 8.19e-03, 9.26e-03, 1.05e-02, 1.19e-02, 1.35e-02, 1.55e-02, 1.76e-02, 2.00e-02, 2.25e-02, 2.53e-02, 2.80e-02, 3.06e-02, 3.28e-02, 3.40e-02, 3.47e-02, 3.45e-02, 3.34e-02, 3.15e-02, 2.94e-02, 2.70e-02, 2.46e-02, 2.25e-02, 2.06e-02, 1.92e-02, 1.80e-02, 1.71e-02, 1.62e-02, 1.55e-02, 1.48e-02, 1.40e-02, 1.32e-02, 1.23e-02, 1.15e-02, 1.07e-02, 9.87e-03, 9.15e-03, 8.51e-03, 7.98e-03, 7.58e-03, 7.27e-03, 6.95e-03, 6.83e-03, 6.70e-03, 6.58e-03, 6.55e-03, 6.51e-03, 6.50e-03, 6.55e-03, 6.55e-03, 6.60e-03, 6.67e-03, 6.84e-03, 6.97e-03, 7.14e-03, 7.32e-03, 7.59e-03, 7.80e-03, 8.08e-03, 8.38e-03, 8.71e-03, 9.03e-03, 9.29e-03, 9.62e-03, 9.90e-03, 1.02e-02, 1.05e-02, 1.08e-02, 1.11e-02, 1.13e-02, 1.16e-02, 1.18e-02, 1.21e-02, 1.23e-02, 1.25e-02, 1.27e-02, 1.30e-02, 1.31e-02, 1.34e-02, 1.35e-02, 1.36e-02, 1.38e-02, 1.40e-02, 1.41e-02, 1.42e-02, 1.43e-02, 1.44e-02, 1.45e-02, 1.46e-02, 1.47e-02, 1.47e-02, 1.48e-02, 1.48e-02, 1.49e-02, 1.49e-02, 1.50e-02, 1.50e-02, 1.51e-02, 1.52e-02, 1.52e-02, 1.52e-02, 1.53e-02, 1.53e-02, 1.53e-02, 1.53e-02, 1.53e-02, 1.54e-02, 1.55e-02, 1.54e-02, 1.54e-02, 1.55e-02, 1.55e-02, 1.55e-02, 1.55e-02, 1.55e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.57e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.55e-02, 1.55e-02, 1.55e-02, 1.54e-02, 1.54e-02, 1.54e-02, 1.54e-02, 1.54e-02, 1.53e-02, 1.53e-02, 1.53e-02, 1.52e-02, 1.52e-02, 1.51e-02, 1.51e-02, 1.50e-02, 1.49e-02, 1.48e-02, 1.49e-02, 1.48e-02, 1.47e-02, 1.47e-02, 1.46e-02, 1.45e-02, 1.44e-02, 1.44e-02, 1.43e-02, 1.42e-02, 1.42e-02, 1.41e-02, 1.40e-02, 1.39e-02, 1.38e-02, 1.37e-02, 1.36e-02, 1.35e-02, 1.34e-02, 1.33e-02, 1.32e-02, 1.31e-02, 1.30e-02, 1.29e-02, 1.28e-02, 1.26e-02, 1.26e-02, 1.24e-02, 1.23e-02, 1.22e-02, 1.20e-02, 1.20e-02, 1.18e-02, 1.17e-02, 1.16e-02, 1.13e-02, 1.12e-02, 1.11e-02, 1.10e-02, 1.08e-02, 1.06e-02, 1.05e-02, 1.03e-02, 1.02e-02, 1.00e-02, 9.84e-03, 9.70e-03, 9.53e-03, 9.37e-03, 9.21e-03, 9.03e-03, 8.87e-03, 8.69e-03, 8.54e-03, 8.37e-03, 8.23e-03, 8.07e-03, 7.86e-03, 7.76e-03, 7.60e-03, 7.42e-03, 7.26e-03, 7.09e-03, 6.92e-03, 6.77e-03, 6.62e-03, 6.49e-03, 6.36e-03, 6.14e-03, 6.05e-03, 5.91e-03, 5.76e-03, 5.62e-03, 5.51e-03, 5.33e-03, 5.24e-03, 5.09e-03, 4.97e-03, 4.80e-03, 4.70e-03, 4.60e-03, 4.43e-03, 4.37e-03, 4.22e-03, 4.12e-03, 4.03e-03, 3.89e-03, 3.78e-03, 3.69e-03, 3.60e-03, 3.50e-03, 3.39e-03, 3.33e-03, 3.18e-03, 3.09e-03, 3.04e-03, 2.94e-03, 2.85e-03, 2.76e-03, 2.68e-03, 2.60e-03, 2.55e-03, 2.48e-03, 2.39e-03, 2.30e-03, 2.28e-03, 2.19e-03, 2.13e-03, 2.05e-03, 1.97e-03, 1.95e-03, 1.89e-03, 1.85e-03, 1.75e-03, 1.69e-03, 1.64e-03, 1.60e-03, 1.55e-03, 1.48e-03, 1.48e-03, 1.43e-03, 1.36e-03, 1.32e-03, 1.27e-03, 1.24e-03, 1.21e-03, 1.18e-03, 1.15e-03, 1.08e-03, 1.05e-03, 1.01e-03, 1.01e-03, 9.70e-04, 9.27e-04, 9.21e-04, 8.63e-04, 8.61e-04, 8.08e-04, 7.79e-04, 7.79e-04, 7.32e-04, 7.30e-04, 7.07e-04, 6.96e-04, 6.54e-04, 6.31e-04, 6.29e-04, 5.80e-04, 5.84e-04, 5.60e-04, 5.27e-04, 5.28e-04, 5.06e-04, 4.89e-04, 4.89e-04, 4.71e-04, 4.51e-04, 4.23e-04, 4.25e-04, 4.16e-04, 3.93e-04, 3.81e-04, 3.67e-04, 3.41e-04, 3.56e-04, 3.28e-04, 3.22e-04, 3.17e-04, 3.10e-04, 3.10e-04, 2.77e-04, 2.76e-04, 2.79e-04, 2.54e-04, 2.64e-04, 2.47e-04, 2.42e-04, 2.32e-04, 2.20e-04, 2.16e-04, 2.13e-04, 2.04e-04, 1.98e-04, 1.86e-04, 1.81e-04, 1.85e-04, 1.82e-04, 1.71e-04, 1.67e-04, 1.53e-04, 1.42e-04, 1.51e-04, 1.48e-04, 1.47e-04, 1.31e-04, 1.34e-04, 1.51e-04, 1.39e-04, 1.23e-04, 1.18e-04, 1.13e-04, 9.86e-05, 1.09e-04, 1.04e-04, 1.03e-04, 9.62e-05, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00};
    constexpr static float white5506K[]{0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 4.39e-05, 4.49e-05, 4.39e-05, 4.53e-05, 4.39e-05, 4.33e-05, 4.69e-05, 4.60e-05, 4.69e-05, 5.00e-05, 5.27e-05, 5.32e-05, 5.56e-05, 5.93e-05, 6.15e-05, 6.45e-05, 7.03e-05, 7.82e-05, 8.49e-05, 9.30e-05, 1.04e-04, 1.14e-04, 1.27e-04, 1.46e-04, 1.68e-04, 1.93e-04, 2.23e-04, 2.57e-04, 2.96e-04, 3.50e-04, 4.04e-04, 4.68e-04, 5.42e-04, 6.31e-04, 7.20e-04, 8.24e-04, 9.46e-04, 1.08e-03, 1.23e-03, 1.39e-03, 1.57e-03, 1.76e-03, 1.98e-03, 2.21e-03, 2.48e-03, 2.76e-03, 3.08e-03, 3.40e-03, 3.79e-03, 4.18e-03, 4.63e-03, 5.09e-03, 5.61e-03, 6.15e-03, 6.75e-03, 7.40e-03, 8.07e-03, 8.82e-03, 9.61e-03, 1.05e-02, 1.14e-02, 1.25e-02, 1.37e-02, 1.50e-02, 1.64e-02, 1.78e-02, 1.93e-02, 2.09e-02, 2.25e-02, 2.39e-02, 2.52e-02, 2.61e-02, 2.69e-02, 2.72e-02, 2.72e-02, 2.69e-02, 2.63e-02, 2.54e-02, 2.45e-02, 2.33e-02, 2.22e-02, 2.10e-02, 1.99e-02, 1.90e-02, 1.81e-02, 1.72e-02, 1.64e-02, 1.57e-02, 1.50e-02, 1.43e-02, 1.36e-02, 1.29e-02, 1.23e-02, 1.17e-02, 1.11e-02, 1.05e-02, 1.00e-02, 9.58e-03, 9.19e-03, 8.86e-03, 8.59e-03, 8.35e-03, 8.17e-03, 8.01e-03, 7.89e-03, 7.81e-03, 7.74e-03, 7.73e-03, 7.71e-03, 7.73e-03, 7.76e-03, 7.81e-03, 7.88e-03, 7.98e-03, 8.09e-03, 8.25e-03, 8.37e-03, 8.54e-03, 8.70e-03, 8.88e-03, 9.07e-03, 9.25e-03, 9.45e-03, 9.65e-03, 9.84e-03, 1.01e-02, 1.03e-02, 1.04e-02, 1.06e-02, 1.08e-02, 1.10e-02, 1.12e-02, 1.14e-02, 1.15e-02, 1.17e-02, 1.19e-02, 1.20e-02, 1.21e-02, 1.23e-02, 1.24e-02, 1.25e-02, 1.27e-02, 1.28e-02, 1.29e-02, 1.30e-02, 1.31e-02, 1.32e-02, 1.33e-02, 1.34e-02, 1.35e-02, 1.36e-02, 1.37e-02, 1.37e-02, 1.38e-02, 1.39e-02, 1.40e-02, 1.40e-02, 1.41e-02, 1.41e-02, 1.42e-02, 1.43e-02, 1.43e-02, 1.44e-02, 1.44e-02, 1.45e-02, 1.45e-02, 1.46e-02, 1.46e-02, 1.47e-02, 1.48e-02, 1.48e-02, 1.48e-02, 1.49e-02, 1.50e-02, 1.50e-02, 1.50e-02, 1.51e-02, 1.51e-02, 1.52e-02, 1.52e-02, 1.53e-02, 1.53e-02, 1.53e-02, 1.54e-02, 1.54e-02, 1.54e-02, 1.55e-02, 1.55e-02, 1.55e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.56e-02, 1.55e-02, 1.55e-02, 1.55e-02, 1.54e-02, 1.54e-02, 1.54e-02, 1.53e-02, 1.53e-02, 1.52e-02, 1.52e-02, 1.51e-02, 1.50e-02, 1.50e-02, 1.49e-02, 1.48e-02, 1.47e-02, 1.46e-02, 1.45e-02, 1.44e-02, 1.43e-02, 1.42e-02, 1.41e-02, 1.40e-02, 1.39e-02, 1.38e-02, 1.36e-02, 1.35e-02, 1.33e-02, 1.32e-02, 1.31e-02, 1.29e-02, 1.28e-02, 1.26e-02, 1.25e-02, 1.23e-02, 1.22e-02, 1.20e-02, 1.18e-02, 1.17e-02, 1.15e-02, 1.13e-02, 1.12e-02, 1.10e-02, 1.08e-02, 1.07e-02, 1.05e-02, 1.03e-02, 1.02e-02, 1.00e-02, 9.84e-03, 9.67e-03, 9.50e-03, 9.34e-03, 9.18e-03, 9.00e-03, 8.84e-03, 8.68e-03, 8.51e-03, 8.34e-03, 8.19e-03, 8.03e-03, 7.86e-03, 7.72e-03, 7.57e-03, 7.42e-03, 7.25e-03, 7.10e-03, 6.96e-03, 6.81e-03, 6.68e-03, 6.52e-03, 6.37e-03, 6.25e-03, 6.10e-03, 5.97e-03, 5.83e-03, 5.70e-03, 5.57e-03, 5.44e-03, 5.31e-03, 5.20e-03, 5.06e-03, 4.94e-03, 4.83e-03, 4.71e-03, 4.60e-03, 4.49e-03, 4.37e-03, 4.27e-03, 4.15e-03, 4.06e-03, 3.96e-03, 3.86e-03, 3.76e-03, 3.67e-03, 3.58e-03, 3.48e-03, 3.40e-03, 3.30e-03, 3.23e-03, 3.14e-03, 3.05e-03, 2.97e-03, 2.90e-03, 2.83e-03, 2.75e-03, 2.69e-03, 2.62e-03, 2.56e-03, 2.49e-03, 2.43e-03, 2.36e-03, 2.30e-03, 2.23e-03, 2.18e-03, 2.12e-03, 2.06e-03, 2.01e-03, 1.95e-03, 1.90e-03, 1.85e-03, 1.80e-03, 1.75e-03, 1.71e-03, 1.65e-03, 1.61e-03, 1.56e-03, 1.52e-03, 1.48e-03, 1.44e-03, 1.40e-03, 1.37e-03, 1.33e-03, 1.30e-03, 1.26e-03, 1.23e-03, 1.19e-03, 1.16e-03, 1.13e-03, 1.11e-03, 1.07e-03, 1.04e-03, 1.01e-03, 9.91e-04, 9.59e-04, 9.37e-04, 9.06e-04, 8.81e-04, 8.59e-04, 8.37e-04, 8.12e-04, 7.89e-04, 7.67e-04, 7.51e-04, 7.28e-04, 7.07e-04, 6.85e-04, 6.72e-04, 6.51e-04, 6.34e-04, 6.17e-04, 6.03e-04, 5.86e-04, 5.70e-04, 5.55e-04, 5.39e-04, 5.21e-04, 5.08e-04, 4.99e-04, 4.82e-04, 4.71e-04, 4.60e-04, 4.45e-04, 4.35e-04, 4.24e-04, 4.20e-04, 4.18e-04, 4.11e-04, 3.98e-04, 3.84e-04, 3.66e-04, 3.54e-04, 3.43e-04, 3.32e-04, 3.26e-04, 3.18e-04, 3.12e-04, 3.02e-04, 2.93e-04, 2.86e-04, 2.74e-04, 2.71e-04, 2.64e-04, 2.55e-04, 2.50e-04, 2.40e-04, 2.35e-04, 2.28e-04, 2.22e-04, 2.18e-04, 2.14e-04, 2.09e-04, 2.06e-04, 2.02e-04, 1.95e-04, 1.89e-04, 1.83e-04, 1.79e-04, 1.74e-04, 1.71e-04, 1.67e-04, 1.61e-04, 1.59e-04, 1.55e-04, 1.51e-04, 1.45e-04, 1.45e-04, 1.42e-04, 1.37e-04, 1.32e-04, 1.32e-04, 1.29e-04, 1.26e-04, 1.24e-04, 1.23e-04, 1.20e-04, 1.18e-04, 1.23e-04, 1.47e-04, 1.77e-04, 1.85e-04, 1.77e-04, 1.45e-04, 1.17e-04, 1.03e-04, 9.52e-05, 9.73e-05, 9.96e-05, 9.66e-05, 9.52e-05, 9.01e-05, 8.20e-05, 8.17e-05, 8.35e-05};


    if(lambda < 360 || lambda >= 830)
        return 0.f;
    return white5506K[lambda - 360];
}
#endif

namespace osc
{

    typedef gdt::LCG<16> Random;

    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;

    /*! per-ray data now captures random number generator, so programs
        can access RNG state */
    struct PRD
    {
        Random random;
        vec3f pixelColor;
        vec3f nextRayOrigin;
        vec3f nextRayDirection;
        vec3f lastGlassNormal;
        vec3f lastGlassRayDir;
        vec3f lastGlassR;
        vec3i ch_triangle_index; //only updated when hit glass for now
        int depth;
        bool isEnd;
        bool anyHitLight;
        int lambda;
        char pathREGEX[maxDepth] = {};
    };

    static __forceinline__ __device__
    void *unpackPointer(uint32_t i0, uint32_t i1)
    {
        const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
        void *ptr = reinterpret_cast<void *>( uptr );
        return ptr;
    }

    static __forceinline__ __device__
    void packPointer(void *ptr, uint32_t &i0, uint32_t &i1)
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T *getPRD()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T *>( unpackPointer(u0, u1));
    }

    //------------------------------------------------------------------------------
    // closest hit and anyhit programs for radiance-type rays.
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    //------------------------------------------------------------------------------

    extern "C" __global__ void __closesthit__shadow()
    {
        /* not going to be used ... */
    }

    struct Onb
    {
        __forceinline__ __device__ Onb(const vec3f &normal)
        {
            m_tangent = vec3f(0.f);
            m_binormal = vec3f(0.f);
            m_normal = vec3f(0.f);

            m_normal = normal;

            if (fabs(m_normal.x) > fabs(m_normal.z))
            {
                m_binormal.x = -m_normal.y;
                m_binormal.y = m_normal.x;
                m_binormal.z = 0;
            }
            else
            {
                m_binormal.x = 0;
                m_binormal.y = -m_normal.z;
                m_binormal.z = m_normal.y;
            }

            m_binormal = normalize(m_binormal);
            m_tangent = cross(m_binormal, m_normal);
        }

        __forceinline__ __device__ void inverse_transform(vec3f &p) const
        {
            p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
        }

        vec3f m_tangent;
        vec3f m_binormal;
        vec3f m_normal;
    };

    static __device__ __inline__ vec3f reflect(vec3f rayDir, vec3f normal)
    {
        return rayDir - 2.f * dot(normal, rayDir) * normal;
    }

    static __device__ __inline__ vec3f refract(vec3f rayDir, vec3f normal, float eta)
    {
        const float k = 1.f - eta * eta * (1.f - dot(normal, rayDir) * dot(normal, rayDir));
        if(k < 0.f)
            return vec3f(0.f);
        else
            return eta * rayDir - (eta * dot(normal, rayDir) + sqrt(k)) * normal;
    }

    static __device__ __inline__ bool refract(vec3f &w_t, vec3f rayDir, vec3f normal, float eta)
{
    const float k = 1.f - eta * eta * (1.f - dot(normal, rayDir) * dot(normal, rayDir));
    if(k < 0.f)
{
    w_t = vec3f(0.f);
    return false;
}
else
{
w_t = eta * rayDir - (eta * dot(normal, rayDir) + sqrt(k)) * normal;
return true;
}
}

static __device__ __inline__ float fresnel(float cos_theta_i, float cos_theta_t, float eta)
{
    const float rs = (cos_theta_i - cos_theta_t * eta) / (cos_theta_i + eta * cos_theta_t);
    const float rp = (cos_theta_i * eta - cos_theta_t) / (cos_theta_i * eta + cos_theta_t);

    return 0.5f * (rs * rs + rp * rp);
}

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, vec3f& p)
{
    // Uniformly sample disk.
    const float r   = sqrtf( u1 );
    const float phi = 2.0f * 3.141592653589793 * u2;
    p.x = r * cosf( phi );
    p.y = r * sinf( phi );

    // Project up to hemisphere.
    p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
}


extern "C" __global__ void __anyhit__radiance() { /*! for this simple example, this will remain empty */ }

extern "C" __global__ void __anyhit__shadow() { /*! not going to be used */ }

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance()
{
    PRD &prd = *getPRD<PRD>();
    // set to constant white as background color
    prd.isEnd = true;
    prd.pixelColor *= missColor;
}

extern "C" __global__ void __miss__shadow()
{
    // we didn't hit anything, so the light is visible
    vec3f &prd = *(vec3f *) getPRD<vec3f>();
    prd = vec3f(0.f);
}

__device__ float get_fresnel_R(int wavelength, const PRD &prd, vec3f &w_t)
{
    float wavelengthIor = cauchyRefractionIndex(wavelength / 1000.f, cauchyB, cauchyC);
    vec3f frontFacedNormal = prd.lastGlassNormal;
    vec3f rayDir = prd.lastGlassRayDir;
    float cos_theta_i = dot(-rayDir, frontFacedNormal);
    float eta;
    if(cos_theta_i > 0.f)
    {
        // Ray is entering
        eta = wavelengthIor;// Note: does not handle nested dielectrics
    }
    else
    {
        // Ray is exiting; apply Beer's Law.
        // This is derived in Shirley's Fundamentals of Graphics book

        eta = 1.f / wavelengthIor;
        cos_theta_i = -cos_theta_i;
        frontFacedNormal = -frontFacedNormal;
    }

    const bool tir = !refract(w_t, rayDir, frontFacedNormal, eta);
    const float cos_theta_t = -dot(frontFacedNormal, w_t);
    const float R = tir ? 1.f : fresnel(cos_theta_i, cos_theta_t, eta);
    return R;
}

__device__ bool init = false;

__device__ int binarySearchWavelengthBoundary(int ix, int iy, int accumID, const PRD &prd, vec3f rayOrigin, const LaunchParams optixLaunchParams, bool findMinWavelengthBoundary) //find max WavelengthBoundary if findMinWavelengthBoundary == false
{
    int searchMin = 380, searchMax = prd.lambda;
    if(!findMinWavelengthBoundary)
    {
        searchMin = prd.lambda;
        searchMax = 780;
    }
    int currentSampledWavelength;
    if(findMinWavelengthBoundary)
    {
        currentSampledWavelength = searchMin;
    }
    else
    {
        currentSampledWavelength = searchMax;
    }

    while(true)
    {
        vec3f w_t;
        float R = get_fresnel_R(currentSampledWavelength, prd, w_t);

        //ray
        if(1 - R <= 0) //full reflect
        {
            if(findMinWavelengthBoundary)
                searchMin = currentSampledWavelength;
            else
                searchMax = currentSampledWavelength;
            continue;
        }

        PRD wavelengthPrd;
        uint32_t u2, u3;
        packPointer(&wavelengthPrd, u2, u3);
        wavelengthPrd.random.init(ix + accumID * optixLaunchParams.frame.size.x,
                                  iy + accumID * optixLaunchParams.frame.size.y);
        wavelengthPrd.depth = 0;
        wavelengthPrd.anyHitLight = false;
        wavelengthPrd.nextRayDirection = vec3f(0.f);
        wavelengthPrd.nextRayOrigin = vec3f(0.f);
        wavelengthPrd.pixelColor = vec3f(1.f);
        // the values we store the PRD pointer in:
        const vec3f w_in = normalize(w_t);
        optixTrace(optixLaunchParams.traversable,
                   rayOrigin,
                   w_in,
                   0.0001f,    // tmin
                   1e20f,  // tmax
                   0.0f,   // rayTime
                   OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_NONE,//OPTIX_RAY_FLAG_NONE,
                   RADIANCE_RAY_TYPE,            // SBT offset
                   RAY_TYPE_COUNT,               // SBT stride
                   RADIANCE_RAY_TYPE,            // missSBTIndex
                   u2, u3);
        if(wavelengthPrd.anyHitLight == findMinWavelengthBoundary)
        {
            searchMax = currentSampledWavelength;  //binary search update
        }
        else
        {
            searchMin = currentSampledWavelength;  //binary search update
        }

        //found boundary
        if(searchMax - searchMin <= 1)
        {
            if(findMinWavelengthBoundary)
                return searchMax;  //since max is always the one that hits the light
            else
                return searchMin;
        }

        currentSampledWavelength = (searchMax + searchMin) / 2;
    }
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame()
{
    //printf("%d\n", __cplusplus);

    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const int accumID = optixLaunchParams.frame.accumID;
    const auto &camera = optixLaunchParams.camera;

    if(accumID == 0 && !init)
    {
        init = true;
        missColor = vec3f(0.f);

#ifdef SPECTRAL_MODE
        initLambda2xyzArray();
#endif
    }

    PRD prd;
    prd.random.init(ix + accumID * optixLaunchParams.frame.size.x,
                    iy + accumID * optixLaunchParams.frame.size.y);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer(&prd, u0, u1);

    int numPixelSamples = NUM_PIXEL_SAMPLES;

    vec3f pixelColor = 0.f;
    for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
    {
        //clear prd
        prd.isEnd = false;
        prd.depth = 0;
        prd.nextRayDirection = vec3f(0.f);
        prd.nextRayOrigin = vec3f(0.f);
        my_strcpy(prd.pathREGEX, "");

        // normalized screen plane position, in [0,1]^2
        const vec2f screen(vec2f(ix + prd.random(), iy + prd.random())
                           / vec2f(optixLaunchParams.frame.size));

        // generate ray direction
        vec3f rayDir = normalize(camera.direction
                                 + (screen.x - 0.5f) * camera.horizontal
                                 + (screen.y - 0.5f) * camera.vertical);
        vec3f rayOrigin = camera.position;

#ifdef SPECTRAL_MODE
        vec3f previousRayDir = rayDir;

        prd.lambda = prd.random() * 400 + 380;
        //prd.pixelColor = XYZ2RGB(lambda2xyz[computeLambda2xyzIndex(prd.lambda)]); //* vec3f(1.0f/0.300985f, 1.0f / 0.274355f, 1.0f / 0.216741f);
        prd.pixelColor = vec3f(1.f);
#else
        prd.pixelColor = vec3f(1.f);
#endif


        for (int i = 0; !prd.isEnd; i++)
        {
            optixTrace(optixLaunchParams.traversable,
                       rayOrigin,
                       rayDir,
                       0.0001f,    // tmin
                       1e20f,  // tmax
                       0.0f,   // rayTime
                       OptixVisibilityMask(255),
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                       RADIANCE_RAY_TYPE,            // SBT offset
                       RAY_TYPE_COUNT,               // SBT stride
                       RADIANCE_RAY_TYPE,            // missSBTIndex
                       u0, u1);



            if(prd.depth >= RRBeginDepth)
            {
                float p = length(prd.pixelColor);
                p = min(p , 1.f);
                if(prd.random() >= p)
                    break;
                prd.pixelColor /= p;
            }
            if(prd.depth >= maxDepth)
                break;

            if(prd.isEnd)
                break;

#ifdef SPECTRAL_MODE
            previousRayDir = rayDir;
#endif
            rayOrigin = prd.nextRayOrigin;
            rayDir = prd.nextRayDirection;
        }

        if(prd.isEnd)
        {
#ifdef SPECTRAL_MODE
    #ifndef USE_NAIVE_SPECTRAL
            if(my_StringCompare(prd.pathREGEX, "DSSL"))
            {
                int maxWavelength = 780, minWavelength = 380;
                float validSampleCount = 1.f;

                minWavelength = binarySearchWavelengthBoundary(ix, iy, accumID, prd, rayOrigin, optixLaunchParams, true); //find Min wavelength using binary search
                maxWavelength = binarySearchWavelengthBoundary(ix, iy, accumID, prd, rayOrigin, optixLaunchParams, false); //find Max wavelength using binary search

                //printf("%d %d %d\n", minWavelength, maxWavelength, prd.lambda);

                vec3f sampledWavedColor = vec3f(0.f);
                int boundaryLength = (maxWavelength - minWavelength + 1);

                int nOfForLoop = maxWavelength - minWavelength + 1;
                for(int i = minWavelength; i <= maxWavelength; i++)
                {
                    vec3f w_t;
                    float R = get_fresnel_R(i, prd, w_t);
        #ifndef VISIBILITY_BIAS_FOR_BOUNDARY_SAMPLING
                    //bias -> don't do visibility test
                    vec3f colorOfThisSample = prd.pixelColor * XYZ2RGB(lambda2xyz[computeLambda2xyzIndex(i)]) * getWhiteWavelengthDistribution(i) * (1.f - R) / (1.f-prd.lastGlassR);
                    vec3f uniformSampleResultOnSpectralBoundary = (colorOfThisSample * boundaryLength) / nOfForLoop;
                    float importantSamplingTerm = (401.f / (float)boundaryLength);
                    sampledWavedColor += colorOfThisSample * 401.f / nOfForLoop;//uniformSampleResultOnSpectralBoundary / importantSamplingTerm;
        #endif
                }
                pixelColor += sampledWavedColor;
                //pixelColor = vec3f(100000.f, 0, 0);
            }
            else
            {
                pixelColor += prd.pixelColor * getWhiteWavelengthDistribution(prd.lambda) * 401.f * XYZ2RGB(lambda2xyz[computeLambda2xyzIndex(prd.lambda)]);
            }
    #else
            pixelColor += prd.pixelColor * getWhiteWavelengthDistribution(prd.lambda) * 401.f * XYZ2RGB(lambda2xyz[computeLambda2xyzIndex(prd.lambda)]);
    #endif
#else
            pixelColor += prd.pixelColor;
#endif
        }

    }
    //pixelColor *= vec3f(1.0f/0.300985f, 1.0f / 0.274355f, 1.0f / 0.216741f);  //rgb color space normalize
    // and write to frame buffer ...
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    if (accumID == 0)
    {
        const int r = int(255.99f * min(pixelColor.x / numPixelSamples, 1.f));
        const int g = int(255.99f * min(pixelColor.y / numPixelSamples, 1.f));
        const int b = int(255.99f * min(pixelColor.z / numPixelSamples, 1.f));

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);
        optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
        optixLaunchParams.frame.accumulateBuffer[fbIndex] = pixelColor / numPixelSamples;
    }
    else
    {
        vec3f prevColor = optixLaunchParams.frame.accumulateBuffer[fbIndex];
        vec3f newColor = prevColor + (((pixelColor / numPixelSamples) - prevColor) / (accumID + 1));

        optixLaunchParams.frame.accumulateBuffer[fbIndex] = newColor;
        const int r = int(255.99f * min(newColor.x, 1.f));
        const int g = int(255.99f * min(newColor.y, 1.f));
        const int b = int(255.99f * min(newColor.z, 1.f));

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);
        optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
        //printf("%d\n", oldG);
    }
}

extern "C" __global__ void __closesthit__radiance()  //diffuse
{
    const TriangleMeshSBTData &sbtData
            = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();
    prd.depth++;

    const TriangleData triangleData(sbtData);

    // start with some ambient term
    //vec3f pixelColor = (0.1f + 0.2f * fabsf(dot(Ns, rayDir))) * diffuseColor;
    vec3f pixelColor = triangleData.diffuseColor;

    const float z1 = prd.random();
    const float z2 = prd.random();
    vec3f w_in = normalize(vec3f(1, 1, 1));
    cosine_sample_hemisphere( z1, z2, w_in );
    Onb onb(triangleData.Ns);
    onb.inverse_transform( w_in );

    prd.nextRayOrigin = triangleData.surfPos + 1e-3f * triangleData.Ns;
    prd.nextRayDirection = w_in;
    prd.pixelColor *= pixelColor;
    my_strcat(prd.pathREGEX, "D");
}

extern "C" __global__ void __closesthit__metal()
{
    const TriangleMeshSBTData &sbtData
            = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    const TriangleData triangleData(sbtData);

    vec3f reflectDirection = reflect(triangleData.rayDir, triangleData.Ns);

    prd.nextRayOrigin = triangleData.surfPos + 1e-3f * triangleData.Ns;
    prd.nextRayDirection = reflectDirection;
    //prd.pixelColor *= sbtData.color;
    //prd.pixelColor = vec3f(1.f, 0.f, 0.f);
    prd.depth++;
    my_strcat(prd.pathREGEX, "M");
}

extern "C" __global__ void __closesthit__glass()
{
    const TriangleMeshSBTData &sbtData
            = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    const TriangleData triangleData(sbtData);

    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    prd.lastGlassRayDir = triangleData.rayDir;

    //if (dot(rayDir, frontFacedNormal) > 0.f) frontFacedNormal = -frontFacedNormal;
    vec3f frontFacedNormal = triangleData.rawNormal;
    prd.lastGlassNormal = frontFacedNormal;

#ifdef SPECTRAL_MODE
    cauchyB = sbtData.refractionIndex;
    float wavelengthIor = cauchyRefractionIndex(prd.lambda / 1000.f, cauchyB, cauchyC);
#else
    float wavelengthIor = sbtData.refractionIndex;
#endif

    float cos_theta_i = dot(-triangleData.rayDir, frontFacedNormal);
    float eta;
    float t_hit = optixGetRayTmax();
    //vec3f extinction(-1 * log(1.f), -1 * log(1.f), -1 * log(1.f));
    vec3f extinction(-1 * log(0.9f), -1 * log(0.9f), -1 * log(0.9f));
    vec3f transmittance(1.f);
    if(cos_theta_i > 0.f)
    {
        // Ray is entering
        eta = wavelengthIor;// Note: does not handle nested dielectrics
    }
    else
    {
        // Ray is exiting; apply Beer's Law.
        // This is derived in Shirley's Fundamentals of Graphics book.
        transmittance = vec3f(expf(-extinction.x * t_hit), expf(-extinction.y * t_hit), expf(-extinction.z * t_hit));

        eta = 1.f / wavelengthIor;
        cos_theta_i = -cos_theta_i;
        frontFacedNormal = -frontFacedNormal;
    }

    vec3f w_t;
    const bool tir = !refract(w_t, triangleData.rayDir, frontFacedNormal, eta);
    const float cos_theta_t = -dot(frontFacedNormal, w_t);
    const float R = tir ? 1.f : fresnel(cos_theta_i, cos_theta_t, eta);

    prd.lastGlassR = R;

    const float z = prd.random();
    if(z <= R)
    {
        //Reflect
        const vec3f w_in = reflect(normalize(triangleData.rayDir), normalize(frontFacedNormal));
        prd.nextRayDirection = w_in;
        my_strcat(prd.pathREGEX, "R");
    }
    else
    {
        //Refract
        const vec3f w_in = w_t;
        prd.nextRayDirection = w_in;
        my_strcat(prd.pathREGEX, "S");
    }
    prd.nextRayOrigin = triangleData.surfPos;
    prd.pixelColor *= transmittance;
    prd.depth++;
}

extern "C" __global__ void __closesthit__light()
{
    const TriangleMeshSBTData &sbtData
            = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    const TriangleData triangleData(sbtData);

#ifndef PARALLEL_LIGHT //area light
    prd.pixelColor *= sbtData.emissionColor;
    prd.depth++;
    prd.isEnd = true;
    my_strcat(prd.pathREGEX, "L");
#else //parallel light

    const vec3f lightSourceDir = -triangleData.Ns;  //the normal of the light plate

    if(dot(triangleData.rayDir, lightSourceDir) > 0.99f)  //not perfect parallel is ok
    {
        prd.pixelColor *= sbtData.emissionColor * 10.f;
        prd.depth++;
        prd.isEnd = true;
        my_strcat(prd.pathREGEX, "L");
    }
    else //not parallel is considered miss
    {
        prd.isEnd = true;
        prd.pixelColor *= missColor;
    }

#endif
}

extern "C" __global__ void __anyhit__light()
{
    const TriangleMeshSBTData &sbtData
            = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    prd.anyHitLight = true;
}

} // ::osc
