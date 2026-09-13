import json, os, math
import numpy as np, trimesh

ROOT = str(__import__('pathlib').Path(__file__).resolve().parent)
D = json.load(open(ROOT + '/link_data.json'))
FR = {k: (np.array(v['p'], float), np.array(v['R'], float)) for k, v in D['frames'].items()}
os.makedirs(ROOT + '/meshes', exist_ok=True)

PETG, TPU, NYLON = 1.27, 1.20, 1.15
FIXED = {'STS3215': 60.0, 'F6700ZZ': 4.3, 'TactileSwitch_6x6': 0.3}
COSMETIC = ('Carapace', 'Shoulder_Boot', 'Femur_Sleeve', 'Tibia_Sleeve', 'Shoulder_Cap',
            'Knee_Cap', 'Claw_Foot')
RUBBER = ('Plunger_Pin', 'Foot_Pad')
METAL = ('F6700ZZ', 'STS3215', 'Horn_', 'Pinion', 'TactileSwitch')

def matspec(b):
    for k, m in FIXED.items():
        if b.startswith(k): return ('fixed', m)
    if b.startswith('Shoulder_Boot') or b.startswith(RUBBER): return ('rho', TPU)
    if b.startswith('Horn_'): return ('rho', NYLON)
    return ('rho', PETG)

def material(b):
    if b.startswith(RUBBER) or b.startswith('Shoulder_Boot'): return 'rubber'
    if b.startswith(METAL): return 'metal'
    if b.startswith(COSMETIC): return 'sandstone'
    return 'sandstone_dark'

LD = {e['name']: e for e in D['links']}
def load(link):
    out = []
    for bd in LD[link]['bodies']:
        safe = bd['body'].replace(' ', '_').replace('(', '').replace(')', '')
        fn = 'meshes/%s__%s.stl' % (link, safe)
        m = trimesh.load(ROOT + '/' + fn, process=True)
        m.merge_vertices(); m.update_faces(m.nondegenerate_faces()); m.fix_normals()
        kind, val = matspec(bd['body'])
        vol = m.volume * 1e6
        rho = (val / vol) if kind == 'fixed' else val
        m.density = rho * 1000.0
        out.append(dict(body=bd['body'], mesh=m, file=fn, mass=(val if kind == 'fixed' else rho * vol) / 1000.0,
                        mat=material(bd['body'])))
    return out

TPL = ['base', 'coxa_l1', 'femur_l1', 'tibia_l1', 'tibia_m0', 'gripper_m0']
PARTS = {t: load(t) for t in TPL}

def inertial(parts, extra=()):
    items = []
    for p in parts:
        mp = p['mesh'].mass_properties
        m = p['mass']
        items.append((m, np.array(mp['center_mass'], float), np.array(mp['inertia'], float) * (m / mp['mass'])))
    items += list(extra)
    M = sum(i[0] for i in items)
    C = sum(i[0] * i[1] for i in items) / M
    I = np.zeros((3, 3))
    for m, c, Ic in items:
        d = c - C
        I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return M, C, I

def box_I(m, lx, ly, lz):
    return m / 12.0 * np.diag([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])

# payload: 330 g battery on the deck, 100 g electronics above the floor
PAYLOAD = [(0.330, np.array([0, 0, 0.064]), box_I(0.330, 0.138, 0.044, 0.027)),
           (0.100, np.array([0, 0, 0.030]), box_I(0.100, 0.080, 0.060, 0.020))]

def hull(parts, exclude=()):
    ms = [p['mesh'] for p in parts if not p['body'].startswith(exclude)]
    h = trimesh.convex.convex_hull(trimesh.util.concatenate(ms))
    return h

HULLS = {'base': hull(PARTS['base']),
         'coxa': hull(PARTS['coxa_l1']),
         'femur': hull(PARTS['femur_l1']),
         'tibia': hull(PARTS['tibia_l1'], exclude=('Claw_Foot', 'Plunger_Pin')),
         'tibia_m': hull(PARTS['tibia_m0']),
         'gripper': hull(PARTS['gripper_m0'], exclude=('Foot_Pad',))}
for k, h in HULLS.items():
    h.export(ROOT + '/meshes/hull_%s.stl' % k)

INR = {'base': inertial(PARTS['base'], PAYLOAD),
       'coxa': inertial(PARTS['coxa_l1']),
       'femur': inertial(PARTS['femur_l1']),
       'tibia': inertial(PARTS['tibia_l1']),
       'tibia_m': inertial(PARTS['tibia_m0']),
       'gripper': inertial(PARTS['gripper_m0'])}

# ---------- stance geometry ----------
LZ = 0.022; RSW = 0.10112712429686843; CL = 0.042; FL = 0.130; TL = 0.170; WX = 0.090
Q_LIFT = math.radians(-50.0); Q_ELB = math.radians(122.0)
GROUND = -0.044730          # world z (CAD frame) of the foot contact in stance
z_elbow = 0.1215858
z_wrist = 0.0359907
R_STD, R_MAN = 0.012, 0.010
c_std = (z_elbow - (GROUND + R_STD)) / math.sin(math.radians(72.0))
c_man = (z_wrist - (GROUND + R_MAN)) / math.sin(math.radians(72.0))
BASE_Z = -GROUND + 0.0003
print('foot sphere std: c=%.5f r=%.3f | manip: c=%.5f r=%.3f | base z=%.5f' % (c_std, R_STD, c_man, R_MAN, BASE_Z))

LIM = dict(sweep=(-math.radians(45), math.radians(45)),
           lift=(math.radians(-80), math.radians(85)),
           elbow=(math.radians(10), math.radians(130)),
           elbow_m=(math.radians(10), math.radians(135)),
           wrist=(-math.radians(75), math.radians(75)))
THETA = [0.0, 72.0, 144.0, -144.0, -72.0]

# --- Feetech STS3215 servo model (12 V, 30 kgf.cm) ---
ROTOR_INERTIA = 5.0e-8          # kg m^2, small brushed rotor (assumed)
GEAR_RATIO = 345.0              # STS3215 reduction
ARMATURE = ROTOR_INERTIA * GEAR_RATIO ** 2      # reflected inertia
EFFORT = 2.9                    # N m stall torque at 12 V
VLIM = 4.7                      # rad/s no-load (0.222 s / 60 deg)
NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0
KP = ARMATURE * NATURAL_FREQ ** 2
KV = 2.0 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ
FRICLOSS, DAMPING = 0.05, 0.01
print('servo: armature=%.6f kp=%.3f kv=%.4f effort=%.2f' % (ARMATURE, KP, KV, EFFORT))

mass_total = INR['base'][0] + 5 * (INR['coxa'][0] + INR['femur'][0]) + 4 * INR['tibia'][0] + INR['tibia_m'][0] + INR['gripper'][0]
print('=== link masses (kg) ===')
for k in INR: print('  %-9s %6.3f  com %s' % (k, INR[k][0], np.round(INR[k][1], 4)))
print('TOTAL %.3f kg' % mass_total)
json.dump({'base_z': BASE_Z, 'c_std': c_std, 'c_man': c_man, 'r_std': R_STD, 'r_man': R_MAN,
           'mass_total': mass_total,
           'links': {k: {'m': INR[k][0], 'com': INR[k][1].tolist(), 'I': INR[k][2].tolist()} for k in INR},
           'hull_faces': {k: len(h.faces) for k, h in HULLS.items()}},
          open(ROOT + '/model_cache.json', 'w'), indent=1)
print('hull faces', {k: len(h.faces) for k, h in HULLS.items()})

# ---------------- simplify hulls ----------------
def coarse_hull(h, grid=0.003):
    q = np.unique(np.round(h.vertices / grid).astype(int), axis=0) * grid
    return trimesh.convex.convex_hull(trimesh.PointCloud(q))
CH = {}
for k, h in HULLS.items():
    c = coarse_hull(h)
    CH[k] = c if len(c.faces) < len(h.faces) else h
    CH[k].export(ROOT + '/meshes/hull_%s.stl' % k)
    print('hull %-8s faces %4d -> %4d   vol %.1f -> %.1f cm3' % (k, len(h.faces), len(CH[k].faces), h.volume*1e6, CH[k].volume*1e6))

# ---------------- emit URDF + MJCF ----------------
LEGKIND = [('0', 'm'), ('1', 's'), ('2', 's'), ('3', 's'), ('4', 's')]
VIS = {'coxa': PARTS['coxa_l1'], 'femur': PARTS['femur_l1'], 'tibia': PARTS['tibia_l1'],
       'tibia_m': PARTS['tibia_m0'], 'gripper': PARTS['gripper_m0'], 'base': PARTS['base']}
def inr_xml_urdf(key):
    m, c, I = INR[key]
    return ('    <inertial>\n      <origin xyz="%.10f %.10f %.10f"/>\n      <mass value="%.10f"/>\n'
            '      <inertia ixx="%.12g" ixy="%.12g" ixz="%.12g" iyy="%.12g" iyz="%.12g" izz="%.12g"/>\n'
            '    </inertial>\n') % (c[0], c[1], c[2], m, I[0,0], I[0,1], I[0,2], I[1,1], I[1,2], I[2,2])
def vis_urdf(key):
    s = ''
    for p in VIS[key]:
        s += ('    <visual>\n      <geometry><mesh filename="%s"/></geometry>\n'
              '      <material name="%s"/>\n    </visual>\n') % (p['file'], p['mat'])
    return s
def col_urdf(key, spheres=()):
    s = ('    <collision>\n      <geometry><mesh filename="meshes/hull_%s.stl"/></geometry>\n    </collision>\n') % key
    for (c, r) in spheres:
        s += ('    <collision>\n      <origin xyz="%.10f 0 0"/>\n      <geometry><sphere radius="%.10f"/></geometry>\n    </collision>\n') % (c, r)
    return s

U = ['<?xml version="1.0"?>', '<robot name="rocky">',
     '  <mujoco><compiler meshdir="" balanceinertia="true" discardvisual="false" strippath="false"/></mujoco>',
     '  <material name="sandstone"><color rgba="0.588 0.455 0.329 1"/></material>',
     '  <material name="sandstone_dark"><color rgba="0.376 0.290 0.212 1"/></material>',
     '  <material name="metal"><color rgba="0.25 0.25 0.27 1"/></material>',
     '  <material name="rubber"><color rgba="0.12 0.11 0.10 1"/></material>']
U.append('  <link name="base_link">\n' + inr_xml_urdf('base') + vis_urdf('base') + col_urdf('base') + '  </link>')
for k, kind in LEGKIND:
    th = math.radians(THETA[int(k)])
    U.append('  <link name="coxa_%s">\n%s%s%s  </link>' % (k, inr_xml_urdf('coxa'), vis_urdf('coxa'), col_urdf('coxa')))
    U.append('  <link name="femur_%s">\n%s%s%s  </link>' % (k, inr_xml_urdf('femur'), vis_urdf('femur'), col_urdf('femur')))
    if kind == 's':
        U.append('  <link name="tibia_%s">\n%s%s%s  </link>' % (k, inr_xml_urdf('tibia'), vis_urdf('tibia'), col_urdf('tibia', [(c_std, R_STD)])))
    else:
        U.append('  <link name="tibia_%s">\n%s%s%s  </link>' % (k, inr_xml_urdf('tibia_m'), vis_urdf('tibia_m'), col_urdf('tibia_m')))
        U.append('  <link name="gripper_%s">\n%s%s%s  </link>' % (k, inr_xml_urdf('gripper'), vis_urdf('gripper'), col_urdf('gripper', [(c_man, R_MAN)])))
    def jt(name, parent, child, xyz, rpy, axis, lo, hi):
        return ('  <joint name="%s" type="revolute">\n    <parent link="%s"/>\n    <child link="%s"/>\n'
                '    <origin xyz="%s" rpy="%s"/>\n    <axis xyz="%s"/>\n'
                '    <limit lower="%.10f" upper="%.10f" effort="%.2f" velocity="%.2f"/>\n'
                '    <dynamics damping="%.4f" friction="%.4f"/>\n  </joint>') % (
                name, parent, child, xyz, rpy, axis, lo, hi, EFFORT, VLIM, DAMPING, FRICLOSS)
    U.append(jt('sweep_%s' % k, 'base_link', 'coxa_%s' % k,
                '%.10f %.10f %.10f' % (RSW*math.cos(th), RSW*math.sin(th), LZ), '0 0 %.10f' % th, '0 0 1', *LIM['sweep']))
    U.append(jt('lift_%s' % k, 'coxa_%s' % k, 'femur_%s' % k, '%.10f 0 0' % CL, '0 0 0', '0 1 0', *LIM['lift']))
    U.append(jt('elbow_%s' % k, 'femur_%s' % k, 'tibia_%s' % k, '%.10f 0 0' % FL, '0 0 0', '0 1 0',
                *(LIM['elbow_m'] if kind == 'm' else LIM['elbow'])))
    if kind == 'm':
        U.append(jt('wrist_%s' % k, 'tibia_%s' % k, 'gripper_%s' % k, '%.10f 0 0' % WX, '0 0 0', '0 1 0', *LIM['wrist']))
U.append('</robot>')
open(ROOT + '/rocky.urdf', 'w').write('\n'.join(U) + '\n')
print('wrote rocky.urdf')

# ---------------- emit MJCF ----------------
def mesh_assets():
    seen, out = set(), []
    for key in ['base', 'coxa', 'femur', 'tibia', 'tibia_m', 'gripper']:
        for p in VIS[key]:
            nm = os.path.basename(p['file'])[:-4]
            if nm in seen: continue
            seen.add(nm)
            out.append('    <mesh name="%s" file="%s"/>' % (nm, os.path.basename(p['file'])))
    for k in CH: out.append('    <mesh name="hull_%s" file="hull_%s.stl"/>' % (k, k))
    return out

def vis_mjcf(key, ind):
    return ['%s<geom class="visual" mesh="%s" material="%s"/>' % (ind, os.path.basename(p['file'])[:-4], p['mat']) for p in VIS[key]]

def inr_mjcf(key, ind):
    m, c, I = INR[key]
    return '%s<inertial pos="%.10f %.10f %.10f" mass="%.10f" fullinertia="%.12g %.12g %.12g %.12g %.12g %.12g"/>' % (
        ind, c[0], c[1], c[2], m, I[0,0], I[1,1], I[2,2], I[0,1], I[0,2], I[1,2])

X = ['<mujoco model="rocky">',
     '  <compiler angle="radian" meshdir="meshes" autolimits="true"/>',
     '',
     '  <default>',
     '    <default class="rocky">',
     '      <joint damping="%.10g" armature="%.10g" frictionloss="%.10g"/>' % (DAMPING, ARMATURE, FRICLOSS),
     '      <geom solref="0.01 1"/>',
     '      <default class="visual">',
     '        <geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/>',
     '      </default>',
     '      <default class="collision">',
     '        <geom group="3" contype="1" conaffinity="1" density="0" friction="1 0.005 0.0005"/>',
     '      </default>',
     '    </default>',
     '  </default>',
     '',
     '  <asset>',
     '    <material name="sandstone" rgba="0.588 0.455 0.329 1" specular="0.15" shininess="0.15"/>',
     '    <material name="sandstone_dark" rgba="0.376 0.290 0.212 1" specular="0.15" shininess="0.15"/>',
     '    <material name="metal" rgba="0.25 0.25 0.27 1" specular="0.6" shininess="0.5"/>',
     '    <material name="rubber" rgba="0.12 0.11 0.10 1" specular="0.05" shininess="0.05"/>'] + mesh_assets() + ['  </asset>', '',
     '  <worldbody>',
     '    <body name="base" pos="0 0 %.10f" childclass="rocky">' % BASE_Z,
     '      <freejoint name="floating_base_joint"/>',
     inr_mjcf('base', '      '),
     '      <site name="imu" pos="0 0 0.0125" size="0.005" group="4"/>']
X += vis_mjcf('base', '      ')
X.append('      <geom name="base_collision" class="collision" type="mesh" mesh="hull_base"/>')
for k, kind in LEGKIND:
    th = math.radians(THETA[int(k)])
    lo, hi = LIM['sweep']
    X += ['',
          '      <body name="coxa_%s" pos="%.10f %.10f %.10f" euler="0 0 %.10f">' % (k, RSW*math.cos(th), RSW*math.sin(th), LZ, th),
          '        <joint name="sweep_%s" axis="0 0 1" range="%.10f %.10f" actuatorfrcrange="-%.2f %.2f"/>' % (k, lo, hi, EFFORT, EFFORT),
          inr_mjcf('coxa', '        '),
          ]
    X += vis_mjcf('coxa', '        ')
    X.append('        <geom name="coxa_%s_collision" class="collision" type="mesh" mesh="hull_coxa"/>' % k)
    lo, hi = LIM['lift']
    X += ['        <body name="femur_%s" pos="%.10f 0 0">' % (k, CL),
          '          <joint name="lift_%s" axis="0 1 0" range="%.10f %.10f" actuatorfrcrange="-%.2f %.2f"/>' % (k, lo, hi, EFFORT, EFFORT),
          inr_mjcf('femur', '          ')]
    X += vis_mjcf('femur', '          ')
    X.append('          <geom name="femur_%s_collision" class="collision" type="mesh" mesh="hull_femur"/>' % k)
    lo, hi = LIM['elbow_m'] if kind == 'm' else LIM['elbow']
    tkey = 'tibia_m' if kind == 'm' else 'tibia'
    X += ['          <body name="tibia_%s" pos="%.10f 0 0">' % (k, FL),
          '            <joint name="elbow_%s" axis="0 1 0" range="%.10f %.10f" actuatorfrcrange="-%.2f %.2f"/>' % (k, lo, hi, EFFORT, EFFORT),
          inr_mjcf(tkey, '            ')]
    X += vis_mjcf(tkey, '            ')
    X.append('            <geom name="tibia_%s_collision" class="collision" type="mesh" mesh="hull_%s"/>' % (k, tkey))
    if kind == 's':
        X += ['            <site name="foot_%s" pos="%.10f 0 0" size="%.10f" group="4"/>' % (k, c_std, R_STD),
              '            <geom name="foot_%s_collision" class="collision" type="sphere" size="%.10f" pos="%.10f 0 0" condim="6" priority="1"/>' % (k, R_STD, c_std),
              '          </body>']
    else:
        lo, hi = LIM['wrist']
        X += ['            <body name="gripper_%s" pos="%.10f 0 0">' % (k, WX),
              '              <joint name="wrist_%s" axis="0 1 0" range="%.10f %.10f" actuatorfrcrange="-%.2f %.2f"/>' % (k, lo, hi, EFFORT, EFFORT),
              inr_mjcf('gripper', '              ')]
        X += vis_mjcf('gripper', '              ')
        X += ['              <geom name="gripper_%s_collision" class="collision" type="mesh" mesh="hull_gripper"/>' % k,
              '              <site name="foot_%s" pos="%.10f 0 0" size="%.10f" group="4"/>' % (k, c_man, R_MAN),
              '              <geom name="foot_%s_collision" class="collision" type="sphere" size="%.10f" pos="%.10f 0 0" condim="6" priority="1"/>' % (k, R_MAN, c_man),
              '            </body>', '          </body>']
    X += ['        </body>', '      </body>']
X += ['    </body>', '  </worldbody>', '',
      '  <sensor>',
      '    <gyro name="imu_ang_vel" site="imu"/>',
      '    <velocimeter name="imu_lin_vel" site="imu"/>',
      '    <accelerometer name="imu_lin_acc" site="imu"/>',
      '    <framezaxis name="imu_upvector" objtype="body" objname="world" reftype="site" refname="imu"/>',
      '    <subtreeangmom name="root_angmom" body="base"/>',
      '  </sensor>', '']
QJ = []
for k, kind in LEGKIND:
    QJ += [0.0, Q_LIFT, Q_ELB] + ([0.0] if kind == 'm' else [])
qpos = '0 0 %.10f 1 0 0 0 ' % BASE_Z + ' '.join('%.10f' % v for v in QJ)
X += ['  <keyframe>', '    <key name="home" qpos="%s"/>' % qpos, '  </keyframe>', '</mujoco>']
open(ROOT + '/rocky.xml', 'w').write('\n'.join(X) + '\n')
print('wrote rocky.xml   nq_joints=%d' % len(QJ))

JOINTS = []
for k, kind in LEGKIND:
    JOINTS += ['sweep_%s' % k, 'lift_%s' % k, 'elbow_%s' % k] + (['wrist_%s' % k] if kind == 'm' else [])
S = ['<mujoco model="rocky_standalone">',
     '  <include file="rocky.xml"/>',
     '  <option timestep="0.005" iterations="10" ls_iterations="20"/>',
     '  <statistic center="0 0 0.1" extent="0.7"/>',
     '  <visual><global offwidth="1024" offheight="768"/><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/><rgba haze="0.15 0.25 0.35 1"/></visual>',
     '  <asset>',
     '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>',
     '    <texture type="2d" name="grid" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300"/>',
     '    <material name="grid" texture="grid" texrepeat="6 6" texuniform="true" reflectance="0.1"/>',
     '  </asset>',
     '  <worldbody>',
     '    <light pos="0 0 2" dir="0 0 -1" directional="true"/>',
     '    <geom name="floor" type="plane" size="0 0 0.05" material="grid" condim="3" friction="1 0.005 0.0005"/>',
     '  </worldbody>',
     '  <actuator>']
for j in JOINTS:
    S.append('    <position name="%s" joint="%s" kp="%.10g" kv="%.10g" forcerange="-%.2f %.2f" ctrlrange="-3.15 3.15"/>' % (j, j, KP, KV, EFFORT, EFFORT))
S += ['  </actuator>', '</mujoco>']
open(ROOT + '/rocky_standalone.xml', 'w').write('\n'.join(S) + '\n')
print('wrote rocky_standalone.xml')
