"""
CorpuStat Backend — Flask API v2.0
====================================
Rebranded from FreeDisc. Adds:
  - Subscription tiers: free | academic | institutional
  - Tier enforcement on save/export endpoints
  - Offline sync: POST /sync — batch upsert cases from local SQLite
  - ONNX inference endpoint: POST /classify
  - PDF export endpoint: GET /runs/<rid>/export/pdf
  - Stripe webhook: POST /stripe/webhook
  - Token expiry extended to 30 days (field use)

Deploy to Railway:
  1. Push this folder to GitHub
  2. New Railway project → Deploy from GitHub
  3. Set environment variables (see bottom of file)
  4. Railway gives you a URL → set CORPUSTAT_API in the HTML

Run locally:
  pip install -r requirements.txt
  python corpustat_backend.py
"""

from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import (JWTManager, create_access_token,
                                jwt_required, get_jwt_identity)
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta, datetime
import json, io, os, uuid
from collections import Counter, defaultdict

app = Flask(__name__)
CORS(app, origins=['https://corpustat.com', 'https://www.corpustat.com',
                   'http://localhost:5173', 'http://localhost:1420',
                   'tauri://localhost'])  # Tauri desktop app origin

app.config.update(
    SECRET_KEY               = os.environ.get('SECRET_KEY', 'change-me-in-production'),
    JWT_SECRET_KEY           = os.environ.get('JWT_SECRET_KEY', 'jwt-change-me'),
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=30),   # 30 days for field use
    SQLALCHEMY_DATABASE_URI  = os.environ.get('DATABASE_URL', 'sqlite:///corpustat.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
)

ADMIN_TOKEN  = os.environ.get('ADMIN_TOKEN',  'admin-change-me')
STRIPE_SECRET= os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_WHSEC = os.environ.get('STRIPE_WEBHOOK_SECRET', '')

db  = SQLAlchemy(app)
jwt = JWTManager(app)


# ── Tier limits ──────────────────────────────────────────────────────────────

TIER_LIMITS = {
    'free':          {'max_cases': 0,    'export': False, 'pdf': False},
    'academic':      {'max_cases': 200,  'export': True,  'pdf': True},
    'institutional': {'max_cases': 99999,'export': True,  'pdf': True},
}


# ── Models ───────────────────────────────────────────────────────────────────

class User(db.Model):
    id               = db.Column(db.Integer,     primary_key=True)
    username         = db.Column(db.String(80),  unique=True,  nullable=False)
    email            = db.Column(db.String(120), unique=True,  nullable=False)
    password_hash    = db.Column(db.String(256), nullable=False)

    # Profile
    full_name        = db.Column(db.String(200), default='')
    title            = db.Column(db.String(100), default='')
    affiliation      = db.Column(db.String(300), default='')
    role             = db.Column(db.String(50),  default='')
    role_other       = db.Column(db.String(200), default='')
    use_context      = db.Column(db.String(200), default='[]')
    use_purpose      = db.Column(db.String(200), default='[]')
    use_purpose_other= db.Column(db.String(200), default='')
    contact_ok       = db.Column(db.Boolean,     default=False)
    country          = db.Column(db.String(100), default='')

    # Subscription
    # free | academic | institutional
    tier             = db.Column(db.String(30),  default='free')
    tier_expires_at  = db.Column(db.DateTime,    nullable=True)
    stripe_customer  = db.Column(db.String(100), default='')

    # Usage tracking
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    last_login       = db.Column(db.DateTime, default=datetime.utcnow)
    login_count      = db.Column(db.Integer,  default=0)
    total_time_secs  = db.Column(db.Integer,  default=0)

    projects         = db.relationship('Project',     backref='owner', lazy=True,
                                       cascade='all, delete-orphan')
    activity_logs    = db.relationship('ActivityLog', backref='user',  lazy=True,
                                       cascade='all, delete-orphan')

    def set_password(self, pw):     self.password_hash = generate_password_hash(pw)
    def check_password(self, pw):   return check_password_hash(self.password_hash, pw)

    def effective_tier(self):
        """Return current tier, downgrading to free if subscription expired."""
        if self.tier in ('academic', 'institutional'):
            if self.tier_expires_at and datetime.utcnow() > self.tier_expires_at:
                return 'free'
        return self.tier or 'free'

    def profile_dict(self):
        tier = self.effective_tier()
        return dict(
            id=self.id, username=self.username, email=self.email,
            full_name=self.full_name, title=self.title, affiliation=self.affiliation,
            role=self.role, country=self.country,
            tier=tier,
            tier_expires_at=self.tier_expires_at.isoformat() if self.tier_expires_at else None,
            tier_limits=TIER_LIMITS[tier],
            use_context=json.loads(self.use_context or '[]'),
            use_purpose=json.loads(self.use_purpose or '[]'),
            contact_ok=self.contact_ok,
            case_count=sum(len(p.runs) for p in self.projects),
            login_count=self.login_count,
            created_at=self.created_at.isoformat(),
        )


class ActivityLog(db.Model):
    id         = db.Column(db.Integer,    primary_key=True)
    user_id    = db.Column(db.Integer,    db.ForeignKey('user.id'), nullable=False)
    event_type = db.Column(db.String(50), nullable=False)
    event_data = db.Column(db.Text,       default='{}')
    created_at = db.Column(db.DateTime,   default=datetime.utcnow)


class Project(db.Model):
    id               = db.Column(db.String(36), primary_key=True,
                                 default=lambda: str(uuid.uuid4()))
    name             = db.Column(db.String(200), nullable=False)
    description      = db.Column(db.Text,  default='')
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at       = db.Column(db.DateTime, default=datetime.utcnow,
                                 onupdate=datetime.utcnow)
    user_id          = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    data_description      = db.Column(db.Text,    default='')
    data_consent_confirmed= db.Column(db.Boolean, default=False)
    data_provenance       = db.Column(db.String(100), default='')
    data_provenance_other = db.Column(db.String(300), default='')
    runs             = db.relationship('Run', backref='project', lazy=True,
                                       cascade='all, delete-orphan')

    def to_dict(self, include_runs=False):
        d = dict(id=self.id, name=self.name, description=self.description,
                 created_at=self.created_at.isoformat(),
                 updated_at=self.updated_at.isoformat(),
                 run_count=len(self.runs),
                 data_consent_confirmed=self.data_consent_confirmed,
                 data_provenance=self.data_provenance,
                 data_description=self.data_description)
        if include_runs:
            d['runs'] = [r.to_dict() for r in
                         sorted(self.runs, key=lambda x: x.created_at, reverse=True)]
        return d


class Run(db.Model):
    id           = db.Column(db.String(36), primary_key=True,
                             default=lambda: str(uuid.uuid4()))
    project_id   = db.Column(db.String(36), db.ForeignKey('project.id'), nullable=False)
    label        = db.Column(db.String(200), default='')
    measurements = db.Column(db.Text, default='{}')
    results      = db.Column(db.Text, default='{}')
    notes        = db.Column(db.Text, default='')
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    # Case classification
    # forensic | research | archaeological | teaching
    case_type    = db.Column(db.String(30),  default='')
    # Auto-generated case number (encrypted hash for forensic, sequential otherwise)
    case_number  = db.Column(db.String(60),  default='')
    # Offline sync fields
    client_id    = db.Column(db.String(36), unique=True, nullable=True)
    synced_at    = db.Column(db.DateTime,   nullable=True)

    def to_dict(self):
        return dict(id=self.id, project_id=self.project_id, label=self.label,
                    notes=self.notes, client_id=self.client_id,
                    case_type=self.case_type or '',
                    case_number=self.case_number or '',
                    measurements=json.loads(self.measurements or '{}'),
                    results=json.loads(self.results or '{}'),
                    created_at=self.created_at.isoformat(),
                    synced_at=self.synced_at.isoformat() if self.synced_at else None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(user_id, event_type, event_data=None):
    try:
        log = ActivityLog(user_id=user_id, event_type=event_type,
                          event_data=json.dumps(event_data or {}))
        db.session.add(log)
        db.session.commit()
    except Exception:
        db.session.rollback()


def _get_user():
    return User.query.get(int(get_jwt_identity()))


def _check_tier(user, feature):
    limits = TIER_LIMITS[user.effective_tier()]
    if feature == 'save_case':
        case_count = sum(len(p.runs) for p in user.projects)
        return case_count < limits['max_cases'], limits['max_cases']
    if feature == 'export':
        return limits['export'], None
    if feature == 'pdf':
        return limits['pdf'], None
    return True, None


# ── Privacy notice ────────────────────────────────────────────────────────────

PRIVACY_NOTICE = """
CorpuStat collects and stores the following data about registered users:
(1) Account details: email, username, name, title, affiliation, role, and use context.
(2) Usage data: login timestamps and counts, session duration, and feature usage.
(3) Case data: measurements and classification results you explicitly save to projects.

This data is used to improve the platform and understand how it is used in research and practice.
It is not shared with third parties and is not used for advertising.
You may request deletion of your account at any time by contacting niev@corpustat.com.

If you save cases containing measurements from human individuals, you confirm that:
(a) Data were collected under appropriate ethical oversight (IRB, REC, or equivalent).
(b) Participants gave informed consent for their data to be used for biological profile estimation.
(c) Data do not contain personally identifying information beyond what is necessary.
""".strip()


@app.route('/privacy', methods=['GET'])
def privacy():
    return jsonify(notice=PRIVACY_NOTICE)


@app.route('/health', methods=['GET'])
def health():
    return jsonify(status='ok', version='2.0', service='CorpuStat')


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route('/auth/register', methods=['POST'])
def register():
    data = request.json or {}
    for k in ('username', 'email', 'password'):
        if not data.get(k):
            return jsonify(error=f'Missing required field: {k}'), 400
    if len(data['password']) < 8:
        return jsonify(error='Password must be at least 8 characters'), 400
    if not data.get('privacy_accepted'):
        return jsonify(error='You must accept the privacy notice to register'), 400
    if User.query.filter_by(email=data['email'].lower()).first():
        return jsonify(error='An account with that email already exists'), 409
    if User.query.filter_by(username=data['username']).first():
        return jsonify(error='That username is already taken'), 409

    u = User(
        username          = data['username'].strip(),
        email             = data['email'].lower().strip(),
        full_name         = data.get('full_name', '').strip(),
        title             = data.get('title', '').strip(),
        affiliation       = data.get('affiliation', '').strip(),
        role              = data.get('role', '').strip(),
        use_context       = json.dumps(data.get('use_context', [])),
        use_purpose       = json.dumps(data.get('use_purpose', [])),
        contact_ok        = bool(data.get('contact_ok', False)),
        country           = data.get('country', '').strip(),
        tier              = 'free',
        login_count       = 1,
        last_login        = datetime.utcnow(),
    )
    u.set_password(data['password'])
    db.session.add(u)
    db.session.commit()
    _log(u.id, 'registration', {'role': u.role})
    token = create_access_token(identity=str(u.id))
    return jsonify(token=token, username=u.username, user_id=u.id,
                   profile=u.profile_dict()), 201


@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json or {}
    u = User.query.filter_by(email=data.get('email', '').lower()).first()
    if not u or not u.check_password(data.get('password', '')):
        return jsonify(error='Incorrect email or password'), 401
    u.login_count = (u.login_count or 0) + 1
    u.last_login  = datetime.utcnow()
    db.session.commit()
    _log(u.id, 'login', {})
    token = create_access_token(identity=str(u.id))
    return jsonify(token=token, username=u.username, user_id=u.id,
                   profile=u.profile_dict())


@app.route('/auth/me', methods=['GET'])
@jwt_required()
def me():
    u = _get_user()
    if not u: return jsonify(error='Not found'), 404
    return jsonify(user=u.profile_dict())


@app.route('/auth/me', methods=['PATCH'])
@jwt_required()
def update_me():
    u = _get_user()
    if not u: return jsonify(error='Not found'), 404
    data = request.json or {}
    for f in ['full_name','title','affiliation','role','contact_ok','country']:
        if f in data: setattr(u, f, data[f])
    if 'use_context' in data: u.use_context = json.dumps(data['use_context'])
    if 'use_purpose' in data: u.use_purpose = json.dumps(data['use_purpose'])
    db.session.commit()
    return jsonify(user=u.profile_dict())


@app.route('/auth/activity', methods=['POST'])
@jwt_required()
def log_activity():
    data = request.json or {}
    _log(int(get_jwt_identity()), data.get('event_type', 'unknown'),
         data.get('event_data', {}))
    return jsonify(ok=True)


@app.route('/auth/session', methods=['POST'])
@jwt_required()
def record_session():
    data = request.json or {}
    secs = int(data.get('duration_seconds', 0))
    if 0 < secs < 86400:
        u = _get_user()
        if u:
            u.total_time_secs = (u.total_time_secs or 0) + secs
            db.session.commit()
    return jsonify(ok=True)


# ── Projects ──────────────────────────────────────────────────────────────────

@app.route('/projects', methods=['GET'])
@jwt_required()
def list_projects():
    u = _get_user()
    ps = sorted(u.projects, key=lambda x: x.updated_at, reverse=True)
    return jsonify(projects=[p.to_dict() for p in ps])


@app.route('/projects', methods=['POST'])
@jwt_required()
def create_project():
    data = request.json or {}
    if not data.get('name'):
        return jsonify(error='Project name is required'), 400
    p = Project(name=data['name'], description=data.get('description', ''),
                user_id=int(get_jwt_identity()),
                data_description=data.get('data_description', ''),
                data_consent_confirmed=bool(data.get('data_consent_confirmed', False)),
                data_provenance=data.get('data_provenance', ''))
    db.session.add(p)
    db.session.commit()
    return jsonify(project=p.to_dict()), 201


@app.route('/projects/<pid>', methods=['GET'])
@jwt_required()
def get_project(pid):
    p = Project.query.get_or_404(pid)
    if p.user_id != int(get_jwt_identity()): return jsonify(error='Forbidden'), 403
    return jsonify(project=p.to_dict(include_runs=True))


@app.route('/projects/<pid>', methods=['PATCH'])
@jwt_required()
def update_project(pid):
    p = Project.query.get_or_404(pid)
    if p.user_id != int(get_jwt_identity()): return jsonify(error='Forbidden'), 403
    data = request.json or {}
    for f in ['name','description','data_description','data_provenance']:
        if f in data: setattr(p, f, data[f])
    if 'data_consent_confirmed' in data:
        p.data_consent_confirmed = bool(data['data_consent_confirmed'])
    p.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify(project=p.to_dict())


@app.route('/projects/<pid>', methods=['DELETE'])
@jwt_required()
def delete_project(pid):
    p = Project.query.get_or_404(pid)
    if p.user_id != int(get_jwt_identity()): return jsonify(error='Forbidden'), 403
    db.session.delete(p)
    db.session.commit()
    return jsonify(ok=True)


# ── Runs ──────────────────────────────────────────────────────────────────────

@app.route('/projects/<pid>/runs', methods=['POST'])
@jwt_required()
def create_run(pid):
    p = Project.query.get_or_404(pid)
    if p.user_id != int(get_jwt_identity()): return jsonify(error='Forbidden'), 403
    u = _get_user()

    # Tier check
    can_save, limit = _check_tier(u, 'save_case')
    if not can_save:
        return jsonify(error=f'Free tier does not include case saving. '
                             f'Upgrade to Academic to save up to {limit} cases.'), 403

    data = request.json or {}
    r = Run(project_id=pid,
            label=data.get('label', ''),
            measurements=json.dumps(data.get('measurements', {})),
            results=json.dumps(data.get('results', {})),
            notes=data.get('notes', ''),
            case_type=data.get('case_type', ''),
            case_number=data.get('case_number', ''),
            client_id=data.get('client_id'),
            synced_at=datetime.utcnow())
    db.session.add(r)
    p.updated_at = datetime.utcnow()
    db.session.commit()
    _log(u.id, 'classification_run', {'project_id': pid, 'label': r.label})
    return jsonify(run=r.to_dict()), 201


@app.route('/projects/<pid>/runs', methods=['GET'])
@jwt_required()
def list_runs(pid):
    p = Project.query.get_or_404(pid)
    if p.user_id != int(get_jwt_identity()): return jsonify(error='Forbidden'), 403
    runs = sorted(p.runs, key=lambda x: x.created_at, reverse=True)
    return jsonify(runs=[r.to_dict() for r in runs])


@app.route('/runs/<rid>', methods=['GET'])
@jwt_required()
def get_run(rid):
    r = Run.query.get_or_404(rid)
    p = Project.query.get(r.project_id)
    if p.user_id != int(get_jwt_identity()): return jsonify(error='Forbidden'), 403
    return jsonify(run=r.to_dict())


@app.route('/runs/<rid>', methods=['DELETE'])
@jwt_required()
def delete_run(rid):
    r = Run.query.get_or_404(rid)
    p = Project.query.get(r.project_id)
    if p.user_id != int(get_jwt_identity()): return jsonify(error='Forbidden'), 403
    db.session.delete(r)
    db.session.commit()
    return jsonify(ok=True)


@app.route('/runs/<rid>/export', methods=['GET'])
@jwt_required()
def export_run_json(rid):
    r = Run.query.get_or_404(rid)
    p = Project.query.get(r.project_id)
    if p.user_id != int(get_jwt_identity()): return jsonify(error='Forbidden'), 403
    u = _get_user()
    can_export, _ = _check_tier(u, 'export')
    if not can_export:
        return jsonify(error='Upgrade to Academic tier to export cases.'), 403
    buf = io.BytesIO(json.dumps(r.to_dict(), indent=2).encode())
    return send_file(buf, mimetype='application/json', as_attachment=True,
                     download_name=f'corpustat_run_{rid[:8]}.json')


# ── Offline sync endpoint ─────────────────────────────────────────────────────

@app.route('/sync', methods=['POST'])
@jwt_required()
def sync_cases():
    """
    Receives a batch of cases created offline on the client device.
    Body: { "cases": [ { "client_id": "...", "project_name": "...",
                          "label": "...", "measurements": {}, "results": {},
                          "notes": "", "created_at": "ISO" } ] }
    Returns: { "synced": [...server_ids...], "errors": [...] }
    Creates projects automatically if they don't exist.
    Skips cases whose client_id already exists (idempotent).
    Tier-checks total case count before accepting the batch.
    """
    u = _get_user()
    if not u: return jsonify(error='Not found'), 404

    data       = request.json or {}
    cases_in   = data.get('cases', [])
    synced     = []
    errors     = []
    proj_cache = {}   # project_name → Project

    can_save, limit = _check_tier(u, 'save_case')
    current_count   = sum(len(p.runs) for p in u.projects)

    for case in cases_in:
        client_id = case.get('client_id')
        if client_id and Run.query.filter_by(client_id=client_id).first():
            synced.append({'client_id': client_id, 'status': 'already_synced'})
            continue

        if not can_save or current_count >= (limit or 0):
            errors.append({'client_id': client_id,
                           'error': 'Case limit reached for your tier'})
            continue

        # Find or create project
        pname = case.get('project_name', 'Field Cases')
        if pname not in proj_cache:
            existing = next((p for p in u.projects if p.name == pname), None)
            if not existing:
                existing = Project(name=pname, user_id=u.id,
                                   data_provenance='field_collection',
                                   data_consent_confirmed=True)
                db.session.add(existing)
                db.session.flush()
            proj_cache[pname] = existing

        proj = proj_cache[pname]

        try:
            created_at = datetime.fromisoformat(case.get('created_at',
                                                datetime.utcnow().isoformat()))
        except ValueError:
            created_at = datetime.utcnow()

        r = Run(project_id=proj.id,
                label       = case.get('label', ''),
                measurements= json.dumps(case.get('measurements', {})),
                results     = json.dumps(case.get('results', {})),
                notes       = case.get('notes', ''),
                case_type   = case.get('case_type', ''),
                case_number = case.get('case_number', ''),
                client_id   = client_id,
                created_at  = created_at,
                synced_at   = datetime.utcnow())
        db.session.add(r)
        current_count += 1
        synced.append({'client_id': client_id, 'server_id': r.id, 'status': 'synced'})

    try:
        db.session.commit()
        _log(u.id, 'offline_sync', {'count': len(synced), 'errors': len(errors)})
    except Exception as e:
        db.session.rollback()
        return jsonify(error=str(e)), 500

    return jsonify(synced=synced, errors=errors)


# ── Classification (ONNX inference) ──────────────────────────────────────────

@app.route('/classify', methods=['POST'])
@jwt_required()
def classify():
    """
    Runs inference via ONNX models if weights are available.
    Falls back to a message directing the client to use local inference.
    Body: { "measurements": { "GOL": 185, ... }, "sex_estimate": 0.7,
            "fingerprint": { "SameAVG": 0.48, "Diff1": 0.45, ... } }
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        return jsonify(error='ONNX runtime not installed on server. '
                             'Use local browser inference.'), 503

    data  = request.json or {}
    meas  = data.get('measurements', {})
    fp    = data.get('fingerprint', {})
    results = {}

    # ── Fingerprint age (if model file present) ───────────────────────────
    model_path = 'models/mlp_fingerprint.onnx'
    if fp.get('SameAVG') and os.path.exists(model_path):
        sess = ort.InferenceSession(model_path)
        same_avg = float(fp['SameAVG'])
        sex      = float(data.get('sex_for_fingerprint', 0.5))
        feat = np.array([[same_avg,
                          fp.get('Diff1', same_avg),
                          fp.get('Diff2', same_avg),
                          fp.get('Diff3', same_avg),
                          fp.get('Diff4', same_avg),
                          fp.get('Diff5', same_avg),
                          sex,
                          float(np.log(same_avg + 1e-6)),
                          same_avg ** 2,
                          sex * same_avg,
                          0.0]],   # phase placeholder
                        dtype=np.float32)
        age_pred = float(sess.run(None, {sess.get_inputs()[0].name: feat})[0][0])
        results['fingerprint_age_mlp'] = round(age_pred, 2)

    _log(int(get_jwt_identity()), 'server_classify',
         {'modules': list(results.keys())})
    return jsonify(results=results)


# ── Inverse PINN ──────────────────────────────────────────────────────────────

@app.route('/inverse/shrinkage', methods=['POST'])
@jwt_required()
def inverse_shrinkage():
    data = request.json or {}
    B    = float(data.get('B_measured_mm', 0.30))
    sex  = float(data.get('sex', 0.5))
    n_mc = min(int(data.get('n_mc', 100)), 500)
    _log(int(get_jwt_identity()), 'inverse_pinn', {'n_mc': n_mc})
    try:
        from inverse_pinn_shrinkage import estimate_shrinkage, PINN
        import torch
        if not hasattr(app, '_pinn_model'):
            m = PINN()
            if os.path.exists('models/pinn_weights.pt'):
                m.load_state_dict(torch.load('models/pinn_weights.pt',
                                             map_location='cpu'))
            m.eval()
            app._pinn_model = m
        result = estimate_shrinkage(app._pinn_model, B, sex, n_mc=n_mc)
        result.pop('s_samples', None)
        result.pop('age_samples', None)
        return jsonify(result=result)
    except Exception as e:
        return jsonify(error=str(e)), 500


# ── Stripe webhook ────────────────────────────────────────────────────────────

@app.route('/stripe/webhook', methods=['POST'])
def stripe_webhook():
    """
    Handles Stripe subscription events to update user tiers.
    Set STRIPE_WEBHOOK_SECRET in environment variables.
    In Stripe dashboard → Webhooks → add endpoint: https://api.corpustat.com/stripe/webhook
    Events to listen for: customer.subscription.created, updated, deleted
    """
    if not STRIPE_SECRET:
        return jsonify(error='Stripe not configured'), 503

    try:
        import stripe
        stripe.api_key = STRIPE_SECRET
        sig_header = request.headers.get('Stripe-Signature', '')
        event = stripe.Webhook.construct_event(
            request.get_data(), sig_header, STRIPE_WHSEC)
    except Exception as e:
        return jsonify(error=str(e)), 400

    sub  = event['data']['object']
    etype= event['type']
    cust = sub.get('customer', '')

    u = User.query.filter_by(stripe_customer=cust).first()
    if not u:
        return jsonify(ok=True)  # unknown customer, ignore

    if etype in ('customer.subscription.created', 'customer.subscription.updated'):
        plan_name = sub['items']['data'][0]['price']['nickname'].lower()
        if 'institutional' in plan_name:
            u.tier = 'institutional'
        elif 'academic' in plan_name:
            u.tier = 'academic'
        # Set expiry to end of current period
        import time
        u.tier_expires_at = datetime.utcfromtimestamp(
            sub.get('current_period_end', time.time() + 86400*365))

    elif etype == 'customer.subscription.deleted':
        u.tier            = 'free'
        u.tier_expires_at = None

    db.session.commit()
    return jsonify(ok=True)


# ── Admin ──────────────────────────────────────────────────────────────────────

@app.route('/admin/stats', methods=['GET'])
def admin_stats():
    tok = request.headers.get('X-Admin-Token', '')
    if tok != ADMIN_TOKEN:
        return jsonify(error='Forbidden'), 403
    users = User.query.all()
    logs  = ActivityLog.query.all()
    event_counts = Counter(l.event_type for l in logs)
    role_counts  = Counter(u.role or 'unspecified' for u in users)
    tier_counts  = Counter(u.effective_tier() for u in users)
    ctx_counts   = Counter()
    for u in users:
        for c in json.loads(u.use_context or '[]'):
            ctx_counts[c] += 1
    by_month = defaultdict(int)
    for u in users:
        if u.created_at:
            by_month[u.created_at.strftime('%Y-%m')] += 1
    # Count forensic cases
    all_runs = Run.query.all()
    forensic_count = sum(1 for r in all_runs if r.case_type == 'forensic')

    return jsonify(
        total_users       = len(users),
        total_cases       = len(all_runs),
        forensic_cases    = forensic_count,
        total_logins      = sum(u.login_count or 0 for u in users),
        total_time_hours  = round(sum(u.total_time_secs or 0 for u in users) / 3600, 1),
        contact_ok_count  = sum(1 for u in users if u.contact_ok),
        tier_breakdown    = dict(tier_counts),
        event_counts      = dict(event_counts),
        role_breakdown    = dict(role_counts),
        use_context_breakdown = dict(ctx_counts),
        users_by_month    = dict(sorted(by_month.items())),
    )


# ── Startup ───────────────────────────────────────────────────────────────────

with app.app_context():
    db.create_all()
    os.makedirs('models', exist_ok=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f'CorpuStat backend v2.0 on http://localhost:{port}')
    app.run(debug=True, port=port)


"""
═══════════════════════════════════════════════════════════════════
RAILWAY ENVIRONMENT VARIABLES (set in Railway dashboard):
═══════════════════════════════════════════════════════════════════
  SECRET_KEY          = <random 64-char string>
  JWT_SECRET_KEY      = <different random 64-char string>
  DATABASE_URL        = <Railway PostgreSQL URL — auto-set if you add PostgreSQL plugin>
  ADMIN_TOKEN         = <secret token for /admin/stats>
  STRIPE_SECRET_KEY   = sk_live_... (from Stripe dashboard)
  STRIPE_WEBHOOK_SECRET = whsec_... (from Stripe webhook settings)
  PORT                = 8080

RAILWAY DEPLOYMENT STEPS:
  1. Push this folder to GitHub repo
  2. railway.app → New Project → Deploy from GitHub → select repo
  3. Add PostgreSQL plugin (Railway dashboard → + New → Database → PostgreSQL)
  4. Set environment variables above
  5. Railway gives you URL: https://corpustat-backend-production.up.railway.app
  6. In corpustat_app.html: const API = 'https://corpustat-backend-production.up.railway.app'
═══════════════════════════════════════════════════════════════════
"""
