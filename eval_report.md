# Schema Field Mapper -- Eval Report

Model provider used for LLM-judge metrics: **anthropic**
Quality-metric engine: **deepeval (GEval)**

## Regression metrics (vs. golden_mapping.json)

- Overall accuracy: **1.0**
- Precision: **1.0**  Recall: **1.0**  F1: **1.0**
- Type-transform agreement: **1.0**

| Table | Correct | Total | Accuracy |
|---|---|---|---|
| emp_master | 18 | 18 | 1.0 |
| dept_info | 7 | 7 | 1.0 |
| locations | 8 | 8 | 1.0 |

No mismatches against golden.

## Quality metrics

- Mean relevance: **0.827**
- Mean faithfulness: **0.755**

| Table | Source field | Destination field | Relevance | Faithfulness |
|---|---|---|---|---|
| emp_master | emp_id | _id | 0.8 | 0.9 |
| emp_master | emp_cd | employeeCode | 0.9 | 0.9 |
| emp_master | f_name | fullName.firstName | 0.9 | 0.9 |
| emp_master | l_name | fullName.lastName | 0.9 | 0.8 |
| emp_master | hire_dt | employment.startDate | 0.9 | 0.4 |
| emp_master | term_dt | employment.endDate | 0.9 | 0.9 |
| emp_master | dept_id | department.departmentId | 0.8 | 0.8 |
| emp_master | mgr_emp_id | employment.managerId | 0.9 | 0.9 |
| emp_master | job_lvl_cd | employment.jobLevel | 0.9 | 0.7 |
| emp_master | base_sal | compensation.baseSalary | 0.9 | 0.7 |
| emp_master | sal_currency | compensation.currency | 0.9 | 0.9 |
| emp_master | work_email | contact.email | 0.2 | 0.7 |
| emp_master | work_phone | contact.phone | 0.7 | 0.3 |
| emp_master | office_loc_id | location.locationId | 0.9 | 0.9 |
| emp_master | is_remote | employment.isRemote | 0.9 | 0.9 |
| emp_master | rec_stat | employment.status | 0.9 | 0.9 |
| emp_master | created_ts | meta.createdAt | 0.9 | 0.7 |
| emp_master | updated_ts | meta.updatedAt | 0.9 | 0.7 |
| dept_info | dept_id | _id | 0.2 | 0.9 |
| dept_info | dept_nm | name | 0.8 | 0.7 |
| dept_info | parent_dept_id | parentDepartmentId | 0.8 | 0.7 |
| dept_info | dept_head_id | headEmployeeId | 0.9 | 0.9 |
| dept_info | cost_ctr_cd | costCenterCode | 0.9 | 0.7 |
| dept_info | dept_stat | isActive | 0.9 | 0.8 |
| dept_info | dept_cd | code | 0.7 | 0.7 |
| locations | loc_id | _id | 0.8 | 0.8 |
| locations | loc_cd | code | 0.8 | 0.2 |
| locations | loc_nm | name | 0.9 | 0.7 |
| locations | city | city | 0.9 | 0.8 |
| locations | state_prov | stateOrProvince | 0.9 | 0.7 |
| locations | country_cd | country | 0.9 | 0.9 |
| locations | postal_cd | postalCode | 0.9 | 0.6 |
| locations | tz_cd | timezone | 0.9 | 0.9 |