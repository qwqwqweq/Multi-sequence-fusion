import os
import logging
import pydicom
import shutil
import numpy as np
from datetime import datetime
import dicom2nifti
import dicom2nifti.settings as settings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import glob

CURRENT_TIME = "2025"
CURRENT_USER = "qwqwqweq"

settings.disable_validate_orthogonal()
settings.disable_validate_slice_increment()

class DicomSeriesCollector:
    def __init__(self):
        self.required_tags = [
            'SeriesInstanceUID',
            'ImagePositionPatient',
            'ImageOrientationPatient',
            'PixelSpacing',
            'SliceThickness',
            'SliceLocation'
        ]

    def _determine_series_type(self, ds: pydicom.dataset.FileDataset) -> Tuple[str, str]:
        identifiers = [
            ds.get('SeriesDescription', ''),
            ds.get('ProtocolName', ''),
            ds.get('SequenceName', ''),
            ds.get('ImageType', [])
        ]

        series_identifier = f"{ds.SeriesInstanceUID}_{getattr(ds, 'SeriesNumber', '0')}"
        combined_text = ' '.join(str(x).lower() for x in identifiers)

        if 't1' in combined_text:
            return 'T1', series_identifier
        elif 't2' in combined_text and 'flair' not in combined_text:
            return 'T2', series_identifier

        return 'Other', series_identifier

    def collect_series(self, patient_dir: str) -> Dict[str, Dict]:
        series_dict = defaultdict(lambda: {'files': [], 'metadata': {}})
        for filepath in glob.glob(os.path.join(patient_dir, '**/*.dcm'), recursive=True):
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)

                if not all(hasattr(ds, tag) for tag in self.required_tags):
                    missing_tags = [tag for tag in self.required_tags if not hasattr(ds, tag)]
                    logging.warning(f"文件缺少必需标签 {filepath}: {missing_tags}")
                    continue

                series_type, series_id = self._determine_series_type(ds)

                if series_type not in ['T1', 'T2']:
                    continue

                if series_id not in series_dict:
                    series_dict[series_id] = {
                        'type': series_type,
                        'files': [],
                        'metadata': {
                            'series_description': getattr(ds, 'SeriesDescription', ''),
                            'protocol_name': getattr(ds, 'ProtocolName', ''),
                            'manufacturer': getattr(ds, 'Manufacturer', ''),
                            'pixel_spacing': getattr(ds, 'PixelSpacing', ''),
                            'slice_thickness': getattr(ds, 'SliceThickness', ''),
                            'series_number': getattr(ds, 'SeriesNumber', ''),
                            'series_uid': ds.SeriesInstanceUID
                        }
                    }

                series_dict[series_id]['files'].append({
                    'path': filepath,
                    'position': ds.ImagePositionPatient,
                    'slice_location': float(ds.SliceLocation),
                    'instance_number': int(getattr(ds, 'InstanceNumber', 0))
                })

            except Exception as e:
                logging.warning(f"无法处理文件 {filepath}: {str(e)}")
                continue

        for series_id, info in series_dict.items():
            logging.info(f"找到序列: {info['type']}, 文件数: {len(info['files'])}")

        return dict(series_dict)

    def _extract_metadata(self, ds: pydicom.dataset.FileDataset) -> Dict:
        return {
            'series_description': getattr(ds, 'SeriesDescription', ''),
            'protocol_name': getattr(ds, 'ProtocolName', ''),
            'manufacturer': getattr(ds, 'Manufacturer', ''),
            'pixel_spacing': getattr(ds, 'PixelSpacing', ''),
            'slice_thickness': getattr(ds, 'SliceThickness', ''),
            'series_number': getattr(ds, 'SeriesNumber', ''),
            'series_uid': ds.SeriesInstanceUID
        }

class SeriesValidator:
    def __init__(self):
        self.min_slices = {'T1': 8, 'T2': 8}

    def validate_series(self, series_info: Dict) -> Tuple[bool, Dict]:
        validation_results = {
            'series_type': series_info['type'],
            'n_files': len(series_info['files']),
            'is_valid': False,
            'errors': [],
            'warnings': []
        }

        if series_info['type'] not in ['T1', 'T2']:
            validation_results['errors'].append("非目标序列类型")
            return False, validation_results

        if len(series_info['files']) < self.min_slices[series_info['type']]:
            validation_results['errors'].append(
                f"切片数量不足: {len(series_info['files'])} < {self.min_slices[series_info['type']]}"
            )
            return False, validation_results

        try:
            sorted_files = sorted(series_info['files'], key=lambda x: x['slice_location'])
            slice_locations = [f['slice_location'] for f in sorted_files]
            gaps = np.diff(slice_locations)

            mean_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            uniformity = std_gap / mean_gap if mean_gap != 0 else float('inf')

            validation_results['spacing_info'] = {
                'mean_gap': float(mean_gap),
                'std_gap': float(std_gap),
                'uniformity': float(uniformity)
            }

            if uniformity > 0.1:
                validation_results['errors'].append(f"切片间距不均匀: {uniformity:.3f}")
                return False, validation_results

        except Exception as e:
            validation_results['errors'].append(f"切片位置验证失败: {str(e)}")
            return False, validation_results

        instance_numbers = [f['instance_number'] for f in series_info['files']]
        if max(instance_numbers) - min(instance_numbers) + 1 != len(instance_numbers):
            validation_results['errors'].append("切片序列不连续")
            return False, validation_results

        validation_results['is_valid'] = True
        return True, validation_results

def setup_logging() -> None:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    log_file = os.path.join(log_dir, f'conversion_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

class NiftiConverter:
    def __init__(self):
        settings.disable_validate_orthogonal()
        settings.disable_validate_slice_increment()

    def convert_series(self, series_info: Dict, output_root: str, patient_name: str) -> Optional[str]:
        try:
            current_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            temp_dir = os.path.join(output_root, f"temp_dcm_{current_time}")
            os.makedirs(temp_dir, exist_ok=True)

            sorted_files = sorted(
                series_info['files'],
                key=lambda x: (x['slice_location'], x['instance_number'])
            )

            for idx, file_info in enumerate(sorted_files):
                src = file_info['path']
                dst = os.path.join(temp_dir, f"slice_{idx:03d}.dcm")
                shutil.copy2(src, dst)

            try:
                series_type = series_info['type']  # 'T1' 或 'T2'
                output_dir = os.path.join(output_root, series_type, patient_name)
                os.makedirs(output_dir, exist_ok=True)

                output_file = os.path.join(
                    output_dir,
                    f"{series_type}_{len(sorted_files)}slices_{current_time}.nii.gz"
                )

                dicom2nifti.dicom_series_to_nifti(
                    temp_dir,
                    output_file,
                    reorient_nifti=True
                )

                logging.info(f"序列类型: {series_type}")
                logging.info(f"输出文件: {output_file}")

                return output_file

            except Exception as e:
                logging.error(f"转换失败: {str(e)}")
                return None

            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

        except Exception as e:
            logging.error(f"处理序列时出错: {str(e)}")
            return None

def validate_series_uniqueness(series_dict: Dict) -> Dict:
    t1_series = {}
    t2_series = {}
    other_series = {}

    for series_id, info in series_dict.items():
        if info['type'] == 'T1':
            t1_series[series_id] = info
        elif info['type'] == 'T2':
            t2_series[series_id] = info
        else:
            other_series[series_id] = info

    validated_series = {}
    if t1_series:
        best_t1 = select_best_series(t1_series, 'T1')
        if best_t1:
            validated_series[best_t1['series_id']] = series_dict[best_t1['series_id']]

    if t2_series:
        best_t2 = select_best_series(t2_series, 'T2')
        if best_t2:
            validated_series[best_t2['series_id']] = series_dict[best_t2['series_id']]

    return validated_series

def select_best_series(series_dict: Dict, series_type: str) -> Optional[Dict]:
    if len(series_dict) > 1:
        logging.warning(f"发现多个{series_type}序列，开始选择最佳序列")

    series_scores = []
    for series_id, info in series_dict.items():
        try:
            orientations = []
            for file_info in info['files']:
                ds = pydicom.dcmread(file_info['path'])
                if hasattr(ds, 'ImageOrientationPatient'):
                    orientations.append(ds.ImageOrientationPatient)
            if not orientations:
                logging.warning(f"序列 {series_id} 缺少方向信息，跳过")
                continue

            ref_orientation = orientations[0]
            is_consistent = all(
                np.allclose(orient, ref_orientation, rtol=1e-3, atol=1e-3)
                for orient in orientations
            )

            if not is_consistent:
                logging.warning(f"序列 {series_id} 的切片方向不一致，跳过")
                continue

            first_file = pydicom.dcmread(info['files'][0]['path'])

            score = {
                'series_id': series_id,
                'slice_count': len(info['files']),
                'orientation_score': 0,
                'quality_score': 0
            }

            orientation = first_file.ImageOrientationPatient
            if len(orientation) == 6:
                if series_type == 'T2':
                    sagittal_ref = [1, 0, 0, 0, 1, 0]
                    transverse_ref = [1, 0, 0, 0, 1, 0]

                    sagittal_similarity = sum(abs(a - b) for a, b in zip(orientation, sagittal_ref))
                    transverse_similarity = sum(abs(a - b) for a, b in zip(orientation, transverse_ref))

 
                    if sagittal_similarity < 0.1:
                        score['orientation_score'] = 3
                    elif transverse_similarity < 0.1:
                        score['orientation_score'] = 2
                    elif sagittal_similarity < 0.5 or transverse_similarity < 0.5:
                        score['orientation_score'] = 1
                else:
                    sagittal_ref = [1, 0, 0, 0, 1, 0]
                    similarity = sum(abs(a - b) for a, b in zip(orientation, sagittal_ref))
                    if similarity < 0.1:
                        score['orientation_score'] = 3
                    elif similarity < 0.5:
                        score['orientation_score'] = 2
                    else:
                        score['orientation_score'] = 1

            if hasattr(first_file, 'ImageType'):
                if 'ORIGINAL' in first_file.ImageType:
                    score['quality_score'] += 1
                if 'PRIMARY' in first_file.ImageType:
                    score['quality_score'] += 1

            series_scores.append(score)
            logging.info(f"序列 {series_id} 评分信息：")
            logging.info(f"- 方向评分：{score['orientation_score']}")
            logging.info(f"- 切片数量：{score['slice_count']}")
            logging.info(f"- 图像质量：{score['quality_score']}")
            if hasattr(first_file, 'SeriesDescription'):
                logging.info(f"- 序列描述：{first_file.SeriesDescription}")

        except Exception as e:
            logging.warning(f"评估序列 {series_id} 时出错: {str(e)}")
            continue

    if series_scores:
        best_series = max(series_scores, key=lambda x: (
            x['orientation_score'],  # 首选合适方向
            x['quality_score'],      # 其次是图像质量
            x['slice_count']         # 最后是切片数量
        ))

        logging.info(f"\n选择的{series_type}序列信息：")
        logging.info(f"- 序列ID：{best_series['series_id']}")
        logging.info(f"- 方向评分：{best_series['orientation_score']}")
        logging.info(f"- 切片数量：{best_series['slice_count']}")
        logging.info(f"- 图像质量：{best_series['quality_score']}")

        return best_series
    else:
        logging.warning(f"未找到有效的{series_type}序列")
        return None


def save_series_as_nifti(series_info: Dict, output_dir: str, series_type: str) -> str:
    try:
        current_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        temp_dir = os.path.join(output_dir, f"temp_dcm_{current_time}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            sorted_files = sorted(
                series_info['files'],
                key=lambda x: (x['slice_location'], x['instance_number'])
            )

            for idx, file_info in enumerate(sorted_files):
                shutil.copy2(
                    file_info['path'],
                    os.path.join(temp_dir, f"slice_{idx:03d}.dcm")
                )

            output_file = os.path.join(
                output_dir,
                f"{series_type}_{len(sorted_files)}slices_{current_time}.nii.gz"
            )

            dicom2nifti.dicom_series_to_nifti(
                temp_dir,
                output_file,
                reorient_nifti=True
            )

            logging.info(f"序列类型: {series_type}")
            logging.info(f"输出文件: {output_file}")

            return output_file

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    except Exception as e:
        logging.error(f"保存{series_type}序列时出错: {str(e)}")
        return None

def process_patient(patient_dir: str, output_root: str) -> Dict:
    patient_name = os.path.basename(patient_dir)
    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"\n开始处理患者: {patient_name} - {current_time}")

    results = {
        'patient_name': patient_name,
        'series_results': {},
        'success': False,
        'processed_time': current_time,
        'processed_by': "qwqwqweq"
    }

    try:
        collector = DicomSeriesCollector()
        series_dict = collector.collect_series(patient_dir)

        if not series_dict:
            logging.warning(f"未找到有效的DICOM序列: {patient_dir}")
            return results

        t1_series = {}
        t2_series = {}
        for series_id, info in series_dict.items():
            if info['type'] == 'T1':
                t1_series[series_id] = info
            elif info['type'] == 'T2':
                t2_series[series_id] = info

        if t1_series:
            if len(t1_series) > 1:
                logging.warning(f"发现多个T1序列，开始选择最佳序列")
                best_t1 = select_best_series(t1_series, 'T1')
                t1_series = {best_t1['series_id']: series_dict[best_t1['series_id']]}
                
            for series_id, info in t1_series.items():
                t1_output_dir = os.path.join(output_root, 'T1', patient_name)
                os.makedirs(t1_output_dir, exist_ok=True)
                output_file = save_series_as_nifti(info, t1_output_dir, 'T1')
                if output_file:
                    # 记录该序列处理成功的信息
                    results['series_results'][series_id] = {
                        'series_type': 'T1',
                        'n_files': len(info['files']),
                        'output_file': output_file,
                        'is_valid': True
                    }
        if t2_series:
            if len(t2_series) > 1:
                logging.warning(f"发现多个T2序列，开始选择最佳序列")
                best_t2 = select_best_series(t2_series, 'T2')
                t2_series = {best_t2['series_id']: series_dict[best_t2['series_id']]}

            for series_id, info in t2_series.items():
                t2_output_dir = os.path.join(output_root, 'T2', patient_name)
                os.makedirs(t2_output_dir, exist_ok=True)
                output_file = save_series_as_nifti(info, t2_output_dir, 'T2')
                if output_file:
                    results['series_results'][series_id] = {
                        'series_type': 'T2',
                        'n_files': len(info['files']),
                        'output_file': output_file,
                        'is_valid': True
                    }

        if results['series_results']:
            results['success'] = True

    except Exception as e:
        logging.error(f"处理患者 {patient_name} 时出错: {str(e)}")

    return results


def generate_report(results: List[Dict], output_path: str):
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(output_path, f'conversion_report_{timestamp}.txt')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("DICOM到NIfTI转换报告\n")
        f.write("="*50 + "\n")
        f.write(f"处理时间 (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总处理患者数: {len(results)}\n")
        f.write(f"成功处理患者数: {sum(1 for r in results if r['success'])}\n\n")

        series_stats = defaultdict(int)
        for result in results:
            for series_info in result['series_results'].values():
                if series_info.get('is_valid'):
                    series_stats[series_info['series_type']] += 1

        f.write("序列统计:\n")
        for series_type, count in series_stats.items():
            f.write(f"{series_type}: {count}\n")
        f.write("\n" + "="*50 + "\n\n")

        for result in results:
            f.write(f"\n患者: {result['patient_name']}\n")
            f.write(f"处理时间: {result['processed_time']}\n")
            f.write("-"*30 + "\n")

            for series_uid, info in result['series_results'].items():
                f.write(f"序列类型: {info['series_type']}\n")
                f.write(f"切片数量: {info['n_files']}\n")

                if 'spacing_info' in info:
                    f.write("切片间距信息:\n")
                    for k, v in info['spacing_info'].items():
                        f.write(f"  {k}: {v:.6f}\n")

                if info.get('errors'):
                    f.write("错误:\n")
                    for error in info['errors']:
                        f.write(f"  - {error}\n")

                if info.get('warnings'):
                    f.write("警告:\n")
                    for warning in info['warnings']:
                        f.write(f"  - {warning}\n")

                if 'output_file' in info:
                    f.write(f"输出文件: {info['output_file']}\n")

                f.write("-"*30 + "\n")

    logging.info(f"\n转换报告已保存至: {report_file}")

def main():
    setup_logging()

    input_paths = ["/data/Dataset_Bone/lumbar_vertebra/in_and_o"]
    output_root = "/home/zzs/code/data/in_and_o"
    
    logging.info(f"开始处理 - 时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"输入路径: {input_paths}")
    logging.info(f"输出路径: {output_root}")

    try:
        results = []
        for input_path in input_paths:
            logging.info(f"\n处理输入目录: {input_path}")

            for root, dirs, _ in os.walk(input_path):
                for d in dirs:
                    if d.startswith('导出的病人信息'):
                        export_dir = os.path.join(root, d)
                        logging.info(f"\n发现导出目录: {export_dir}")

                        for patient in os.listdir(export_dir):
                            patient_dir = os.path.join(export_dir, patient)
                            if os.path.isdir(patient_dir):
                                result = process_patient(patient_dir, output_root)
                                results.append(result)

        generate_report(results, output_root)

        success_count = sum(1 for r in results if r['success'])
        logging.info(f"\n处理完成！")
        logging.info(f"总患者数: {len(results)}")
        logging.info(f"成功处理: {success_count}")
        logging.info(f"失败数量: {len(results) - success_count}")

    except Exception as e:
        logging.error(f"执行过程中出错: {str(e)}")
    finally:
        logging.info(f"处理结束时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
